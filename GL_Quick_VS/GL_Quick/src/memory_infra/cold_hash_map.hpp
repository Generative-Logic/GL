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

#include "deload_stats.hpp"
#include "dirty_state.hpp"
#include "lb_arena.hpp"
#include "paged_hash_index.hpp"
#include "paged_vector.hpp"
#include "scratch_arena.hpp"
#include "str_ops.hpp"
#include "infra/rt_tracker.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <type_traits>
#include <vector>

namespace gl {

    /// @brief Location of one cold string in a byte-key store's paged byte
    ///        pool: the logical byte index where its bytes begin, the byte
    ///        length, and the key's cached FNV-1a digest.
    ///
    /// @details
    /// Element type of a `BytesKeyStore`'s id → location index. `byteStart` is
    /// the physical char index in the byte pool (`PagedVector<char>`) where the
    /// string begins; a string occupies `[byteStart, byteStart + len)` and is
    /// interned WITHIN A SINGLE PAGE (no straddle — the pool carries page-tail
    /// padding the deload never emits). Trivially copyable — the index is itself
    /// a paged container element. `hash` caches the full FNV-1a digest beside
    /// the location so lookup never re-walks stored bytes merely to place or
    /// reject a candidate. An interned empty string has `len` zero (no bytes;
    /// `byteStart` is the pool size at intern time).
    ///
    /// The name keeps the `ColdString*` spelling the byte-key store inherited
    /// from the hand-rolled `ColdStringTable` it generalizes, so the seven
    /// `LbMemory` interners and their tests compile unchanged through the
    /// `using ColdStringTable = ColdHashSet<BytesKeyStore>` alias.
    struct ColdStringLocation {
        int32_t byteStart;
        int32_t len;
        uint64_t hash;
    };

    /// @brief Key-store policy for variable-length byte keys — the cold storage
    ///        of a byte-key `ColdHashSet` (the string interner).
    ///
    /// @details
    /// One of the two key-store policies the cold-map family is parameterized
    /// over (the other is `PodKeyStore<K>`, for fixed trivially-copyable keys).
    /// A key store owns ONLY the COLD key columns — paged containers on the LB's
    /// bump arena that deload with it — and supplies the hash + byte-equality the
    /// owning `ColdHashSet`'s heap index probes with. It holds NO index itself:
    /// the "index heap, data cold" split puts the lookup index on the heap inside
    /// `ColdHashSet`, the key DATA cold here.
    ///
    /// `BytesKeyStore` is the storage half of the former `ColdStringTable`,
    /// lifted out verbatim:
    ///
    /// 1. **Bytes** — a `PagedVector<char>` byte pool holding every key's bytes
    ///    in id order (append-only — interner semantics never erase a single
    ///    key; the only resets are the wholesale `clear` / `release`). Each key
    ///    is interned WITHIN A SINGLE PAGE (`appendRunNoStraddle` pads a page
    ///    tail rather than straddle), so it reads back as one contiguous span;
    ///    the deload streams the LOGICAL per-key bytes (the page-tail padding is
    ///    never emitted), so the on-disk image stays dense and byte-identical to
    ///    the pre-paging form.
    /// 2. **Location index** — `PagedVector<ColdStringLocation>`, position
    ///    `id - 1` holds key `id` (id 0 is the reserved null/absent id). Each
    ///    record caches the key's full FNV-1a digest. The digest is runtime
    ///    derived state: canonical deload still emits only lengths + logical
    ///    bytes, and reload recomputes it from those bytes.
    ///
    /// Determinism: appending the same key sequence yields the same arena layout
    /// and the same locations; reload re-bumps element-by-element in id order, so
    /// every (offset, len, hash) reproduces exactly and ids stay valid across
    /// the deload round trip (I-107).
    ///
    /// Threading: `appendKey` is single-threaded write side only (I-83); the
    /// hash / equality / decode reads are safe from the parallel burst.
    ///
    /// @invariant Append-only: an id, once minted, resolves to the same bytes
    ///            for the store's lifetime (until `clear` / `release`, which
    ///            invalidate ALL ids wholesale).
    /// @see `ColdHashSet`, `PagedVector`, `hashSpan`,
    ///      D-165.
    class BytesKeyStore {
    public:
        /// @brief Probe type the owning set's `mint` / `lookup` accept (a byte
        ///        span).
        using KeyView = StrSpan;

        /// @brief Decode result — a zero-copy span over the cold key bytes,
        ///        stable while the LB is resident.
        using KeyDecode = StrSpan;

        /// @brief Number of deload tags this key store contributes: a lengths
        ///        column tag + a content-bytes tag.
        static constexpr int kTagCount = 2;

        /// @brief Bind the cold key columns to the owning LB's arena and the
        ///        aggregate's shared dirty flag.
        ///
        /// @details
        /// Lazy like every paged container: constructing the store consumes no
        /// pool blocks. The dirty flag is escalated by content mutations
        /// (`appendKey`), never by the owning set's index maintenance.
        ///
        /// @param arena The LB's bump arena; outlives the store.
        /// @param dirty The aggregate's shared content-change state.
        BytesKeyStore(LbArena* arena, DirtyState* dirty)
            : locations_(arena, dirty), bytePool_(arena, dirty) {
            assert(arena != nullptr && dirty != nullptr);
        }

        BytesKeyStore(const BytesKeyStore&) = delete;
        BytesKeyStore& operator=(const BytesKeyStore&) = delete;

        /// @brief Number of keys stored (ids run 1..count()).
        ///
        /// @return Key count; asserts residency like every statified size read.
        int32_t count() const { return locations_.size(); }

        /// @brief Whether no key has been stored.
        ///
        /// @return `true` when `count() == 0`.
        bool empty() const { return locations_.empty(); }

        /// @brief Approximate live byte footprint (key content + location
        ///        index).
        ///
        /// @return Live bytes the keys and their location index occupy.
        int64_t liveBytes() const {
            return bytePool_.liveBytes() + locations_.liveBytes();
        }

        /// @brief Return the exact logical key-byte total without walking keys.
        ///
        /// @details
        /// Tracks the sum of every live key length independently of the padded
        /// no-straddle byte pool. Append, canonical reload, copy, compaction, clear,
        /// and release maintain the counter in lockstep, so capacity planning can
        /// read the canonical bytes-tag size in constant time.
        ///
        /// @return Sum of live key lengths in id order.
        /// @invariant Equals `contentBytesFrom(0)` and excludes page-tail padding
        ///            and erased-key holes.
        int64_t logicalByteCount() const { return logicalByteCount_; }

        /// @brief Append `s` at the next id — the store's write primitive.
        ///
        /// @details
        /// Single-threaded write side only (I-83). Copies the bytes into the
        /// LB's arena WITHIN A SINGLE PAGE (`appendRunNoStraddle` pads a page
        /// tail rather than cross it; the empty key adds no bytes, so its
        /// `byteStart` is the current pool size and `len` 0), then records the
        /// location — escalating the shared dirty flag through the location
        /// append. The new id is `count()` after the call (the owning set reads
        /// it). Does NOT touch the heap index — the set inserts into it.
        ///
        /// @param s Key bytes to store.
        void appendKey(const StrSpan& s) {
            assert(s.len >= 0);
            assert(logicalByteCount_
                <= std::numeric_limits<int64_t>::max() - s.len);
            const int32_t start = bytePool_.appendRunNoStraddle(s.ptr, s.len);
            locations_.push_back(ColdStringLocation{
                start, s.len, hashSpan(s) });
            logicalByteCount_ += s.len;
        }

        /// @brief Cached FNV-1a 64-bit hash of stored key `id`'s bytes — the
        ///        hash the owning set's index places stored ids with.
        ///
        /// @details
        /// Read directly from the static paged location record. `appendKey`
        /// computes it once; `copyKeysFrom` copies it; canonical reload
        /// recomputes it from the serialized logical bytes. It uses the same
        /// `hashSpan` digest as a contiguous probe, which is the dedup
        /// correctness condition.
        ///
        /// @param id A stored id (`1 <= id <= count()`).
        /// @return The key's FNV-1a digest.
        uint64_t hashStored(int32_t id) const {
            return locations_[id - 1].hash;
        }

        /// @brief FNV-1a 64-bit hash of a contiguous probe — the hash the owning
        ///        set's index probes with (identical digest to `hashStored` for
        ///        equal bytes).
        ///
        /// @param s The probe bytes.
        /// @return The probe's FNV-1a digest.
        uint64_t hashProbe(const StrSpan& s) const { return hashSpan(s); }

        /// @brief Byte equality of stored key `id` against a contiguous probe,
        ///        with a cached-hash rejection before page resolution.
        ///
        /// @param id        A stored id (the stored side is page-contained).
        /// @param probe     The contiguous lookup bytes.
        /// @param probeHash The probe's already-computed FNV-1a digest.
        /// @return `true` when lengths and bytes match.
        GL_FORCEINLINE bool equalStored(int32_t id, const StrSpan& probe,
                         uint64_t probeHash) const {
            const ColdStringLocation& loc = locations_[id - 1];
            if (loc.hash != probeHash) return false;
            if (loc.len != probe.len) return false;
            if (loc.len == 0) return true;
            int32_t pos = loc.byteStart;
            const int32_t end = loc.byteStart + loc.len;
            int32_t probeOff = 0;
            while (pos < end) {
                int32_t run = 0;
                const char* p = bytePool_.contiguousRun(pos, run);
                if (run > end - pos) run = end - pos;
                if (std::memcmp(p, probe.ptr + probeOff,
                                static_cast<size_t>(run)) != 0)
                    return false;
                pos += run;
                probeOff += run;
            }
            return true;
        }

        /// @brief Decode to a contiguous view of the cold bytes.
        ///
        /// @details
        /// One location read + one page resolve. Every cold key is interned
        /// WITHIN A SINGLE PAGE (`appendRunNoStraddle` skips a page tail rather
        /// than straddle), so a contiguous view always exists — the run-length
        /// assert is the no-straddle guard, not a fast path with a fallback. The
        /// view is stable while the LB is resident and must not be held across a
        /// deload point (I-111).
        ///
        /// @param id A stored id; `1 <= id <= count()`.
        /// @return Contiguous span over the key's bytes; the empty span for an
        ///         interned empty key.
        GL_FORCEINLINE StrSpan decodeAt(int32_t id) const {
            assert(id >= 1 && id <= locations_.size()
                && "BytesKeyStore::decodeAt on an unstored id");
            const ColdStringLocation& loc = locations_[id - 1];
            if (loc.len == 0) return StrSpan();
            int32_t run = 0;
            const char* p = bytePool_.contiguousRun(loc.byteStart, run);
            assert(run >= loc.len
                && "BytesKeyStore::decodeAt: key straddles a page — the "
                   "no-straddle interning invariant is broken");
            return StrSpan(p, loc.len);
        }

        /// @brief Decode to an owned heap copy — the boundary materialization
        ///        (file IO, debug dumps, string-keyed probes).
        ///
        /// @param id A stored id; `1 <= id <= count()`.
        /// @return Heap copy of the bytes (page-aware — walks the byte pool, so
        ///         it works for page-straddling keys where `decodeAt` cannot).
        std::string decodeStringAt(int32_t id) const {
            assert(id >= 1 && id <= locations_.size()
                && "BytesKeyStore::decodeStringAt on an unstored id");
            const ColdStringLocation& loc = locations_[id - 1];
            if (loc.len == 0) return std::string();
            std::string out;
            out.reserve(static_cast<size_t>(loc.len));
            int32_t pos = loc.byteStart;
            const int32_t end = loc.byteStart + loc.len;
            while (pos < end) {
                int32_t run = 0;
                const char* p = bytePool_.contiguousRun(pos, run);
                if (run > end - pos) run = end - pos;
                out.append(p, static_cast<size_t>(run));
                pos += run;
            }
            return out;
        }

        /// @brief Wholesale reset to empty — the owning set's `resetToFresh`.
        ///
        /// @details
        /// Clears the location index (escalating the shared dirty flag) and the
        /// byte pool; the bytes become arena holes the copying compaction
        /// reclaims. Every previously stored id is invalid afterwards. The heap
        /// index is the owning set's to drop.
        void clear() {
            locations_.clear();
            bytePool_.clear();
            logicalByteCount_ = 0;
        }

        /// @brief Drop everything including page capacity — the owning set's
        ///        `release`.
        void release() {
            locations_.release();
            bytePool_.release();
            logicalByteCount_ = 0;
        }

        /// @brief Slide stored key `srcId` down to 0-based position `destPos` —
        ///        the compacting `HashMap::eraseIf` "move a survivor down"
        ///        primitive for the byte-key store.
        ///
        /// @details
        /// Copies ONLY the location entry (byteStart, len); the survivor's bytes
        /// stay where they were interned (no-straddle, so still one contiguous
        /// run), and the erased keys' bytes become holes in the append-only byte
        /// pool that the copying compaction (`LbMemory::reshuffle`) reclaims —
        /// the deload streams the survivors' LOGICAL bytes in id order
        /// (`appendContentBytes` walks per location), so the holes never reach
        /// the canonical image. `destPos <= srcId - 1`. Single-threaded write
        /// side only (I-83).
        ///
        /// @param destPos Destination position in `[0, count())`.
        /// @param srcId   Source id; `1 <= srcId <= count()`.
        void moveKeyTo(int32_t destPos, int32_t srcId) {
            assert(destPos >= 0 && destPos < locations_.size());
            assert(srcId >= 1 && srcId <= locations_.size());
            logicalByteCount_ -= locations_[destPos].len;
            logicalByteCount_ += locations_[srcId - 1].len;
            assert(logicalByteCount_ >= 0);
            locations_.setAt(destPos, locations_[srcId - 1]);
        }

        /// @brief Drop the location index's tail to `newCount` keys — the
        ///        compacting erase's truncate step.
        ///
        /// @details
        /// Truncates ONLY the location index; the byte pool keeps its pages (a
        /// dropped key's bytes are an interior hole the copying compaction
        /// reclaims, not a tail the pool can roll back, since survivors and dead
        /// keys interleave in it). The deload image stays dense — the per-key
        /// logical-byte walk skips the holes. Single-threaded write side only
        /// (I-83).
        ///
        /// @param newCount Retained key count, in `[0, count()]`.
        void truncate(int32_t newCount) {
            assert(newCount >= 0 && newCount <= locations_.size());
            for (int32_t index = newCount; index < locations_.size(); ++index)
                logicalByteCount_ -= locations_[index].len;
            assert(logicalByteCount_ >= 0);
            locations_.truncate(newCount);
        }

        /// @brief Cross-arena deep copy of the keys — the owning set's
        ///        `copyFrom` (the LB-clone path).
        ///
        /// @details
        /// Re-appends `other`'s keys in id order into THIS store's arena,
        /// reproducing ids 1..count() exactly (the empty interned key reproduces
        /// as the same null-offset sentinel). The owning set rebuilds its index
        /// afterwards. Asserts this store is empty (clones copy into fresh
        /// state).
        ///
        /// @param other Source store; resident.
        void copyKeysFrom(const BytesKeyStore& other) {
            assert(locations_.empty()
                && "BytesKeyStore::copyKeysFrom into a non-empty store");
            assert(logicalByteCount_ == 0);
            const int32_t n = other.count();
            for (int32_t id = 1; id <= n; ++id) {
                const ColdStringLocation& oloc = other.locations_[id - 1];
                int32_t start;
                if (oloc.len > 0) {
                    // The source is no-straddle, so its key is one contiguous
                    // run; re-append it no-straddle into this arena (reproducing
                    // the same per-key page layout deterministically).
                    int32_t run = 0;
                    const char* p =
                        other.bytePool_.contiguousRun(oloc.byteStart, run);
                    assert(run >= oloc.len
                        && "BytesKeyStore::copyKeysFrom source straddles a "
                           "page — the no-straddle invariant is broken");
                    start = bytePool_.appendRunNoStraddle(p, oloc.len);
                } else {
                    start = bytePool_.size();
                }
                locations_.push_back(ColdStringLocation{
                    start, oloc.len, oloc.hash });
                logicalByteCount_ += oloc.len;
            }
            assert(logicalByteCount_ == other.logicalByteCount_);
        }

        /// @brief Canonical content dump, lengths part: every key's length, in
        ///        id order, appended to `out` as int32 little endian.
        ///
        /// @details
        /// Together with `appendContentBytes` this is the store's pure-content
        /// image (arena layout invisible — I-103):
        /// the two GLDL tags a byte-key set contributes.
        ///
        /// @param out     Byte sink (the deload chunk buffer).
        /// @param fromRow First key index to append (tail-delta support);
        ///                `0 <= fromRow <= count()`.
        void appendLengthBytes(std::vector<char>& out, int32_t fromRow) const {
            const int32_t n = locations_.size();
            assert(fromRow >= 0 && fromRow <= n);
            for (int32_t i = fromRow; i < n; ++i) {
                const int32_t len = locations_[i].len;
                const char* p = reinterpret_cast<const char*>(&len);
                out.insert(out.end(), p, p + sizeof(int32_t));
            }
        }

        /// @brief Canonical content dump, bytes part: the key bytes concatenated
        ///        in id order, no separators.
        ///
        /// @details
        /// Walks per key in id order: the pool carries page-tail padding (the
        /// no-straddle gaps), so the LOGICAL concatenation — what the image must
        /// be (I-103) — is the per-key bytes only, not
        /// the raw pool. Each key is one page-resident run; the inner loop runs
        /// once per key but stays general.
        ///
        /// @param out     Byte sink.
        /// @param fromRow First key index to append.
        void appendContentBytes(std::vector<char>& out, int32_t fromRow) const {
            const int32_t n = locations_.size();
            assert(fromRow >= 0 && fromRow <= n);
            for (int32_t i = fromRow; i < n; ++i) {
                const ColdStringLocation& loc = locations_[i];
                int32_t pos = loc.byteStart;
                const int32_t end = loc.byteStart + loc.len;
                while (pos < end) {
                    int32_t run = 0;
                    const char* p = bytePool_.contiguousRun(pos, run);
                    if (run > end - pos) run = end - pos;
                    out.insert(out.end(), p, p + run);
                    pos += run;
                }
            }
        }

        /// @brief Total content bytes from `fromRow` to the end — the elemCount
        ///        of the bytes tag.
        ///
        /// @param fromRow First key index counted.
        /// @return Sum of the counted keys' lengths.
        int64_t contentBytesFrom(int32_t fromRow) const {
            const int32_t n = locations_.size();
            assert(fromRow >= 0 && fromRow <= n);
            if (fromRow == 0) return logicalByteCount_;
            int64_t total = 0;
            for (int32_t i = fromRow; i < n; ++i) total += locations_[i].len;
            return total;
        }

        /// @brief Canonical reload: append `rowCount` keys from a lengths array
        ///        + concatenated bytes (the inverse of the two append dumps).
        ///        Does NOT rebuild the owning set's index — the set does.
        ///
        /// @details
        /// Re-bumps element-by-element in id order and recomputes every cached
        /// digest from the canonical bytes, so every (offset, len, hash) — and
        /// therefore every id — reproduces exactly. Loud on malformed input via
        /// asserts.
        ///
        /// @param lengths  `rowCount` int32 lengths in id order.
        /// @param bytes    The concatenated key bytes.
        /// @param byteLen  Total length of `bytes`; must equal the lengths' sum.
        /// @param rowCount Number of keys to append.
        void bulkLoadKeys(const int32_t* lengths, const char* bytes,
                          int64_t byteLen, int32_t rowCount) {
            assert(rowCount >= 0);
            assert(rowCount == 0 || lengths != nullptr);
            assert(byteLen == 0 || bytes != nullptr);
            int64_t off = 0;
            for (int32_t r = 0; r < rowCount; ++r) {
                const int32_t len = lengths[r];
                assert(len >= 0 && "BytesKeyStore::bulkLoadKeys: negative len");
                assert(off + len <= byteLen
                    && "BytesKeyStore::bulkLoadKeys: bytes shorter than "
                       "lengths");
                const int32_t start = (len > 0)
                    ? bytePool_.appendRunNoStraddle(bytes + off, len)
                    : bytePool_.size();
                locations_.push_back(ColdStringLocation{
                    start, len,
                    hashSpan(StrSpan(bytes + off, len)) });
                off += len;
            }
            assert(off == byteLen
                && "BytesKeyStore::bulkLoadKeys: bytes longer than lengths");
            assert(logicalByteCount_
                <= std::numeric_limits<int64_t>::max() - byteLen);
            logicalByteCount_ += byteLen;
        }

        /// @brief Stage one file set's lengths column — the first half of the
        ///        two-tag reload handshake.
        ///
        /// @details
        /// Asserts no stale stash (every staged column must be consumed by the
        /// same set's bytes tag before the next set arrives).
        ///
        /// @param bytes    `rowCount * 4` bytes of int32 lengths.
        /// @param rowCount Number of keys in this set's window.
        void stageLengths(const char* bytes, int64_t rowCount) {
            assert(stagedLengths_.empty()
                && "stageLengths over an unconsumed stash — a file set's bytes "
                   "tag went missing");
            assert(rowCount >= 0);
            stagedLengths_.resize(static_cast<size_t>(rowCount));
            if (rowCount > 0) {
                assert(bytes != nullptr);
                std::memcpy(stagedLengths_.data(), bytes,
                            static_cast<size_t>(rowCount) * sizeof(int32_t));
            }
        }

        /// @brief Consume the staged lengths with this set's content bytes —
        ///        the second half of the reload handshake. Does NOT rebuild the
        ///        owning set's index.
        ///
        /// @param bytes     The concatenated key bytes.
        /// @param byteCount Total content bytes; must equal the staged lengths'
        ///                  sum (asserted in `bulkLoadKeys`).
        void consumeStagedLengthsKeys(const char* bytes, int64_t byteCount) {
            bulkLoadKeys(stagedLengths_.data(), bytes, byteCount,
                         static_cast<int32_t>(stagedLengths_.size()));
            stagedLengths_.clear();
        }

        /// @brief Content dump from a BYTE offset — the bytes tag's tail-window
        ///        form (`deloadedCounts` records the bytes tag's element count,
        ///        which is a byte total).
        ///
        /// @details
        /// Walks the lengths to the window's key boundary (appended-only windows
        /// always start at one — a misaligned offset asserts) and streams from
        /// there.
        ///
        /// @param out      Byte sink.
        /// @param fromByte Byte offset of the window start.
        void appendContentBytesFromByte(std::vector<char>& out,
                                        int64_t fromByte) const {
            assert(fromByte >= 0);
            const int32_t n = locations_.size();
            int64_t acc = 0;
            int32_t i = 0;
            while (i < n && acc < fromByte) {
                acc += locations_[i].len;
                ++i;
            }
            assert(acc == fromByte
                && "bytes tail window does not start at a key boundary");
            appendContentBytes(out, i);
        }

    private:
        // id -> (byteStart, len); position id-1. Paged like every container.
        PagedVector<ColdStringLocation> locations_;

        // Id-order byte pool; a key is the within-page run
        // [byteStart, byteStart+len) (no straddle — page tails may be padded).
        // The bytes dump emits the LOGICAL per-key bytes, not this pool's raw
        // bytes, so the padding never reaches the image.
        PagedVector<char> bytePool_;

        // Exact canonical bytes-tag size. The raw pool is not equivalent because
        // no-straddle placement leaves page-tail padding and erase leaves holes.
        int64_t logicalByteCount_{ 0 };

        // Reload handshake stash: the lengths facet's column waiting for the
        // bytes facet's content (heap bookkeeping; empty outside a load).
        std::vector<int32_t> stagedLengths_;
    };

    /// @brief Key-store policy for fixed trivially-copyable keys — the cold
    ///        storage of a POD-key `ColdHashSet` / `ColdHashMap` / `ColdMultiMap`
    ///        (the shape the int-keyed per-LB maps want).
    ///
    /// @details
    /// The second key-store policy beside `BytesKeyStore`. A fixed key needs no
    /// byte pool and no length column — it is one flat `PagedVector<K>`, position
    /// `id - 1` holding key `id`. Hash and equality run over the key's raw
    /// `sizeof(K)` bytes (`hashSpan`, `std::memcmp`), so the key MUST have a
    /// unique object representation — a `static_assert` rejects padded types,
    /// whose indeterminate padding bytes would make the hash / equality / deload
    /// non-deterministic.
    ///
    /// One deload tag (the dense key column), versus `BytesKeyStore`'s two
    /// (lengths + bytes) — the variable tag count per key store is why the family
    /// lets each store declare `kTagCount`.
    ///
    /// @invariant Append-only: an id, once minted, resolves to the same key for
    ///            the store's lifetime (until `clear` / `release`).
    /// @see `BytesKeyStore`, `ColdHashSet`, `PagedVector`,
    ///      D-165.
    template <typename K>
    class PodKeyStore {
        static_assert(std::is_trivially_copyable<K>::value,
            "PodKeyStore key must be trivially copyable - raw bytes are paged "
            "and streamed to SSD on deload");
        static_assert(std::has_unique_object_representations<K>::value,
            "PodKeyStore key must have a unique object representation - padding "
            "bytes would make the hash / equality / deload non-deterministic");

    public:
        /// @brief Probe type the owning set's `mint` / `lookup` accept.
        using KeyView = K;

        /// @brief Decode result — a const reference into the cold key column,
        ///        stable while the LB is resident.
        using KeyDecode = const K&;

        /// @brief Number of deload tags this key store contributes: one dense
        ///        key column.
        static constexpr int kTagCount = 1;

        /// @brief Bind the cold key column to the owning LB's arena and the
        ///        aggregate's shared dirty flag.
        ///
        /// @param arena The LB's bump arena; outlives the store.
        /// @param dirty The aggregate's shared content-change state.
        PodKeyStore(LbArena* arena, DirtyState* dirty) : keys_(arena, dirty) {
            assert(arena != nullptr && dirty != nullptr);
        }

        PodKeyStore(const PodKeyStore&) = delete;
        PodKeyStore& operator=(const PodKeyStore&) = delete;

        /// @brief Number of keys stored (ids run 1..count()).
        ///
        /// @return Key count.
        int32_t count() const { return keys_.size(); }

        /// @brief Whether no key has been stored.
        ///
        /// @return `true` when `count() == 0`.
        bool empty() const { return keys_.empty(); }

        /// @brief Approximate live byte footprint (the key column).
        ///
        /// @return Live bytes the keys occupy.
        int64_t liveBytes() const { return keys_.liveBytes(); }

        /// @brief Append `k` at the next id — the store's write primitive.
        ///
        /// @details
        /// Single-threaded write side only (I-83). Escalates the shared dirty
        /// flag through the column append. The new id is `count()` after the
        /// call. Does NOT touch the heap index — the owning set inserts into it.
        ///
        /// @param k Key to store.
        void appendKey(const K& k) { keys_.push_back(k); }

        /// @brief FNV-1a 64-bit hash of stored key `id`'s raw bytes — equal to
        ///        `hashProbe` of an equal key.
        ///
        /// @param id A stored id (`1 <= id <= count()`).
        /// @return The key's FNV-1a digest.
        uint64_t hashStored(int32_t id) const {
            const K& k = keys_[id - 1];
            return hashSpan(StrSpan(reinterpret_cast<const char*>(&k),
                                    static_cast<int32_t>(sizeof(K))));
        }

        /// @brief FNV-1a 64-bit hash of a probe key's raw bytes.
        ///
        /// @param k The probe key.
        /// @return The probe's FNV-1a digest.
        uint64_t hashProbe(const K& k) const {
            return hashSpan(StrSpan(reinterpret_cast<const char*>(&k),
                                    static_cast<int32_t>(sizeof(K))));
        }

        /// @brief Raw-byte equality of stored key `id` against a probe.
        ///
        /// @param id    A stored id.
        /// @param probe     The probe key.
        /// @param probeHash The owning lookup's hash (unused: POD equality is a
        ///                  direct fixed-width byte comparison).
        /// @return `true` when the raw bytes match.
        GL_FORCEINLINE bool equalStored(int32_t id, const K& probe, uint64_t probeHash) const {
            (void)probeHash;
            return std::memcmp(&keys_[id - 1], &probe, sizeof(K)) == 0;
        }

        /// @brief Decode an id back to its key (a const reference into the cold
        ///        column).
        ///
        /// @param id A stored id; `1 <= id <= count()`.
        /// @return Const reference to the key.
        GL_FORCEINLINE const K& decodeAt(int32_t id) const {
            assert(id >= 1 && id <= keys_.size()
                && "PodKeyStore::decodeAt on an unstored id");
            return keys_[id - 1];
        }

        /// @brief Overwrite the key at 0-based position `i` — the in-place key
        ///        write the set-map's run-aware compaction (`HashMap::eraseSetIf`)
        ///        slides survivors with.
        ///
        /// @details
        /// Routes through `PagedVector::setAt` (in-place, dirty-marking). The
        /// generic `HashMap::eraseIf` compacts through `moveKeyTo` instead (a
        /// store-position move, so the byte-key store can share the same erase).
        /// Single-threaded write side only (I-83).
        ///
        /// @param i Position in `[0, count())`.
        /// @param k The key to store.
        void setKeyAt(int32_t i, const K& k) { keys_.setAt(i, k); }

        /// @brief Slide stored key `srcId` down to 0-based position `destPos` —
        ///        the compacting `HashMap::eraseIf` "move a survivor down"
        ///        primitive, position-based so both key stores share it.
        ///
        /// @details
        /// `destPos <= srcId - 1`, so the in-place write never overlaps the
        /// source slot. Single-threaded write side only (I-83).
        ///
        /// @param destPos Destination position in `[0, count())`.
        /// @param srcId   Source id; `1 <= srcId <= count()`.
        void moveKeyTo(int32_t destPos, int32_t srcId) {
            keys_.setAt(destPos, keys_[srcId - 1]);
        }

        /// @brief Drop the key column's tail to `newCount` keys — the compacting
        ///        erase's truncate step.
        ///
        /// @param newCount Retained key count, in `[0, count()]`.
        void truncate(int32_t newCount) { keys_.truncate(newCount); }

        /// @brief Wholesale reset to empty — the owning set's `resetToFresh`.
        void clear() { keys_.clear(); }

        /// @brief Drop everything including page capacity — the owning set's
        ///        `release`.
        void release() { keys_.release(); }

        /// @brief Cross-arena deep copy of the keys — the owning set's
        ///        `copyFrom`.
        ///
        /// @param other Source store; resident.
        void copyKeysFrom(const PodKeyStore& other) {
            assert(keys_.empty()
                && "PodKeyStore::copyKeysFrom into a non-empty store");
            keys_ = other.keys_;   // PagedVector deep copy
        }

        /// @brief Dump the dense key column from `fromRow` — the store's single
        ///        deload tag (one `memcpy` per page span).
        ///
        /// @param out     Byte sink.
        /// @param fromRow First key index; `0 <= fromRow <= count()`.
        void appendKeyBytes(std::vector<char>& out, int32_t fromRow) const {
            keys_.appendSpanBytes(out, fromRow);
        }

        /// @brief Bulk-append `rowCount` keys from a dense byte stream — the
        ///        reload path. The owning set rebuilds its index afterwards.
        ///
        /// @param bytes    Source stream of `rowCount * sizeof(K)` bytes.
        /// @param rowCount Keys to append; >= 0.
        void bulkLoadKeyBytes(const char* bytes, int64_t rowCount) {
            keys_.bulkAppendBytes(bytes, rowCount);
        }

    private:
        // id -> key; position id-1. Flat, dense, paged.
        PagedVector<K> keys_;
    };

    /// @brief Value-store policy for a SET — no value column.
    ///
    /// @details
    /// The first of the three value-store policies the unified `HashMap` is
    /// parameterized over (beside `SingleValueStore<V>` and `CsrValueStore<V>`).
    /// A set carries keys only, so this policy holds no buffers and contributes
    /// zero deload tags; its uniform-contract methods are defined no-ops the
    /// `HashMap` engine calls unconditionally (no `if constexpr`). It is an empty
    /// class, so `HashMap`'s private empty-base inheritance elides it entirely —
    /// a `HashMap<KeyStore, EmptyValueStore>` (the `ColdHashSet` alias) keeps the
    /// byte-for-byte layout the hand-rolled set had.
    ///
    /// @see `HashMap`, `SingleValueStore`, `CsrValueStore`,
    ///      D-165.
    class EmptyValueStore {
    public:
        /// @brief Deload tags a set's value side contributes: none.
        static constexpr int kTagCount = 0;

        /// @brief Bind to the LB arena + dirty flag (ignored — a set holds no
        ///        value column; the parameters keep the uniform policy ctor so
        ///        `HashMap` constructs every value store the same way).
        EmptyValueStore(LbArena*, DirtyState*) {}

        EmptyValueStore(const EmptyValueStore&) = delete;
        EmptyValueStore& operator=(const EmptyValueStore&) = delete;

        /// @brief Uniform contract: clear the value column — a no-op for a set.
        void clearValues() {}

        /// @brief Uniform contract: release the value column — a no-op for a
        ///        set.
        void releaseValues() {}

        /// @brief Uniform contract: deep-copy the value column — a no-op for a
        ///        set.
        void copyValuesFrom(const EmptyValueStore&) {}

        /// @brief Uniform contract: value-column live bytes — zero for a set.
        ///
        /// @return 0.
        int64_t valuesLiveBytes() const { return 0; }

        /// @brief Uniform contract: move a value during compaction — a no-op for
        ///        a set (no value column).
        void moveValue(int32_t, int32_t) {}

        /// @brief Uniform contract: truncate the value column — a no-op for a set.
        void truncateValues(int32_t) {}
    };

    /// @brief Value-store policy for a single value per key — one parallel cold
    ///        value column.
    ///
    /// @details
    /// `values_[id - 1]` is key `id`'s value, the same dense paged stream every
    /// other cold container deloads. Set-once: the owning `HashMap::insert`
    /// asserts the key is new and appends one value, so the column grows in
    /// key-id order and the deload bytes are a pure function of the insertion
    /// content. One deload tag (the value column).
    ///
    /// @see `HashMap`, `EmptyValueStore`, `CsrValueStore`,
    ///      D-165.
    template <typename V>
    class SingleValueStore {
    public:
        /// @brief The stored value type (the owning `HashMap`'s value type).
        using ValueType = V;

        /// @brief Deload tags this value side contributes: one value column.
        static constexpr int kTagCount = 1;

        /// @brief Bind the value column to the LB arena + dirty flag.
        ///
        /// @param arena The LB's bump arena; outlives the store.
        /// @param dirty The aggregate's shared content-change state.
        SingleValueStore(LbArena* arena, DirtyState* dirty)
            : values_(arena, dirty) {}

        SingleValueStore(const SingleValueStore&) = delete;
        SingleValueStore& operator=(const SingleValueStore&) = delete;

        /// @brief Uniform contract: clear the value column.
        void clearValues() { values_.clear(); }

        /// @brief Uniform contract: release the value column.
        void releaseValues() { values_.release(); }

        /// @brief Uniform contract: deep-copy the value column (cross-arena).
        ///
        /// @param other Source store; resident.
        void copyValuesFrom(const SingleValueStore& other) {
            values_ = other.values_;
        }

        /// @brief Uniform contract: value-column live bytes.
        ///
        /// @return Live bytes the values occupy.
        int64_t valuesLiveBytes() const { return values_.liveBytes(); }

        /// @brief Append one value at the next id — the `insert` value half.
        ///
        /// @param v The value to store.
        void appendValue(const V& v) { values_.push_back(v); }

        /// @brief Pointer to value `id` (the `find` hit path).
        ///
        /// @param id A stored id; `1 <= id <= size`.
        /// @return Pointer to the value (stable while resident).
        const V* valuePtr(int32_t id) const { return &values_[id - 1]; }

        /// @brief Value at `id`.
        ///
        /// @param id A stored id; `1 <= id <= size`.
        /// @return Const reference to the value.
        const V& valueAt(int32_t id) const { return values_[id - 1]; }

        /// @brief Move key `srcId`'s value to `destId` (1-based) — the compacting
        ///        erase's "slide a survivor's value down" primitive.
        ///
        /// @details
        /// Routes through `PagedVector::setAt`; `destId <= srcId`, so the in-place
        /// write never overlaps the source slot.
        ///
        /// @param destId Destination id.
        /// @param srcId  Source id.
        void moveValue(int32_t destId, int32_t srcId) {
            values_.setAt(destId - 1, values_[srcId - 1]);
        }

        /// @brief Drop the value column's tail to `newCount` values — the
        ///        compacting erase's truncate step.
        ///
        /// @param newCount Retained value count.
        void truncateValues(int32_t newCount) { values_.truncate(newCount); }

        /// @brief In-place overwrite of key `id`'s value — the map's update door
        ///        the set-once `insert` forbids.
        ///
        /// @details
        /// Routes through `PagedVector::setAt`, which marks the aggregate
        /// `Restructured` so the next deload is a full canonical rewrite (an
        /// in-place write is not a tail-delta-eligible append). The first consumer
        /// is `upsertStatementKey`'s flag-bit OR into an existing row.
        ///
        /// @param id A stored id; `1 <= id <= size`.
        /// @param v  The new value.
        void setValueAt(int32_t id, const V& v) { values_.setAt(id - 1, v); }

        /// @brief In-place value overwrite that does NOT escalate the dirty
        ///        state — the parallel-safe twin of `setValueAt` for a map on a
        ///        never-deloaded pool.
        ///
        /// @details
        /// Routes through `PagedVector::setAtRelaxed`: writes only the value slot,
        /// leaving the shared dirty flag untouched, so concurrent calls to
        /// DISJOINT ids are race-free. Sanctioned ONLY for a map on a
        /// never-deloaded pool whose dirty state is meaningless — the pull-model
        /// mail cursor (advanced disjointly in the parallel phase-1 pull). See
        /// `PagedVector::setAtRelaxed` for the full contract.
        ///
        /// @param id A stored id; `1 <= id <= size`.
        /// @param v  The new value.
        void setValueAtRelaxed(int32_t id, const V& v) {
            values_.setAtRelaxed(id - 1, v);
        }

        /// @brief Dump the dense value column from `fromRow` — the value tag.
        ///
        /// @param out     Byte sink.
        /// @param fromRow First value index; `0 <= fromRow <= size`.
        void appendValueBytes(std::vector<char>& out, int32_t fromRow) const {
            values_.appendSpanBytes(out, fromRow);
        }

        /// @brief Bulk-append `rowCount` values from a dense byte stream — the
        ///        value-column reload path.
        ///
        /// @param bytes    Source stream of `rowCount * sizeof(V)` bytes.
        /// @param rowCount Values to append; >= 0.
        void bulkLoadValueBytes(const char* bytes, int64_t rowCount) {
            values_.bulkAppendBytes(bytes, rowCount);
        }

    private:
        PagedVector<V> values_;   // values_[id-1] is key id's value
    };

    /// @brief Value-store policy for an ordered run of values per key, in
    ///        compressed-sparse-row (CSR) form.
    ///
    /// @details
    /// `runStarts_[id - 1]` is where key `id`'s values begin in the flat
    /// `values_` column; a run length is DERIVED (the next key's start, or the
    /// total value count for the last key, minus this key's start). The owning
    /// `HashMap::appendToTail` opens a run for a brand-new key and appends to the
    /// last key's run — a pure append, never an interior shift. Two deload tags
    /// (the run-start column + the value column).
    ///
    /// @see `HashMap`, `EmptyValueStore`, `SingleValueStore`,
    ///      D-165.
    template <typename V>
    class CsrValueStore {
    public:
        /// @brief The stored value type (the owning `HashMap`'s value type).
        using ValueType = V;

        /// @brief Deload tags this value side contributes: run-starts + values.
        static constexpr int kTagCount = 2;

        /// @brief Bind the run-start index + value column to the LB arena +
        ///        dirty flag.
        ///
        /// @param arena The LB's bump arena; outlives the store.
        /// @param dirty The aggregate's shared content-change state.
        CsrValueStore(LbArena* arena, DirtyState* dirty)
            : runStarts_(arena, dirty), values_(arena, dirty) {}

        CsrValueStore(const CsrValueStore&) = delete;
        CsrValueStore& operator=(const CsrValueStore&) = delete;

        /// @brief Uniform contract: clear the run-start + value columns.
        void clearValues() { runStarts_.clear(); values_.clear(); }

        /// @brief Uniform contract: release the run-start + value columns.
        void releaseValues() { runStarts_.release(); values_.release(); }

        /// @brief Uniform contract: deep-copy both columns (cross-arena).
        ///
        /// @param other Source store; resident.
        void copyValuesFrom(const CsrValueStore& other) {
            runStarts_ = other.runStarts_;
            values_ = other.values_;
        }

        /// @brief Uniform contract: value-side live bytes (runs + values).
        ///
        /// @return Live bytes the runs index and values occupy.
        int64_t valuesLiveBytes() const {
            return runStarts_.liveBytes() + values_.liveBytes();
        }

        /// @brief Open a fresh run for a brand-new key at the current value end.
        void openRun() { runStarts_.push_back(values_.size()); }

        /// @brief Append one value to the current tail run.
        ///
        /// @param v The value to append.
        void appendValue(const V& v) { values_.push_back(v); }

        /// @brief Total number of values across every run.
        ///
        /// @return Value count.
        int32_t valueCount() const { return values_.size(); }

        /// @brief Number of values in key `id`'s run — DERIVED from the CSR
        ///        offsets.
        ///
        /// @param id       A stored id; `1 <= id <= keyCount`.
        /// @param keyCount The owning map's key count (to bound the last run).
        /// @return The run length.
        int32_t runLen(int32_t id, int32_t keyCount) const {
            const int32_t start = runStarts_[id - 1];
            const int32_t end = (id < keyCount) ? runStarts_[id]
                                                : values_.size();
            return end - start;
        }

        /// @brief Value `j` of key `id`'s run.
        ///
        /// @param id A stored id.
        /// @param j  Position within the run.
        /// @return Const reference to the value.
        const V& valueAt(int32_t id, int32_t j) const {
            return values_[runStarts_[id - 1] + j];
        }

        /// @brief Dump the dense run-start column from `fromRow` — the runs tag.
        ///
        /// @param out     Byte sink.
        /// @param fromRow First key index; `0 <= fromRow <= keyCount`.
        void appendRunStartBytes(std::vector<char>& out, int32_t fromRow) const {
            runStarts_.appendSpanBytes(out, fromRow);
        }

        /// @brief Bulk-append `rowCount` run-start offsets — the runs reload
        ///        path.
        ///
        /// @param bytes    Source stream of `rowCount * 4` bytes.
        /// @param rowCount Run starts to append; >= 0.
        void bulkLoadRunStartBytes(const char* bytes, int64_t rowCount) {
            runStarts_.bulkAppendBytes(bytes, rowCount);
        }

        /// @brief Dump the dense value column from `fromValue` — the values tag.
        ///
        /// @param out       Byte sink.
        /// @param fromValue First value index; `0 <= fromValue <= valueCount()`.
        void appendValueBytes(std::vector<char>& out, int32_t fromValue) const {
            values_.appendSpanBytes(out, fromValue);
        }

        /// @brief Bulk-append `rowCount` values — the value-column reload path.
        ///
        /// @param bytes    Source stream of `rowCount * sizeof(V)` bytes.
        /// @param rowCount Values to append; >= 0.
        void bulkLoadValueBytes(const char* bytes, int64_t rowCount) {
            values_.bulkAppendBytes(bytes, rowCount);
        }

    private:
        PagedVector<int32_t> runStarts_;   // id-1 -> value-start offset (CSR)
        PagedVector<V> values_;            // all runs concatenated, key-id order
    };

    /// @brief Value-store policy for a SORTED-UNIQUE SET of values per key, in
    ///        compressed-sparse-row (CSR) form — the Batch-2 substrate.
    ///
    /// @details
    /// Same physical layout as `CsrValueStore` (a `runStarts_` index + a flat
    /// `values_` column, `runStarts_[id - 1]` the start of key `id`'s run), but
    /// the run is maintained SORTED and DUPLICATE-FREE by the owning `HashMap`'s
    /// set surface (`insertSorted` / `assignSet`), so it models a
    /// `std::map<K, std::set<V>>` rather than the bag a `CsrValueStore` multimap
    /// holds. The ordering predicate is NOT stored — it is passed per call to
    /// `insertSorted` (the caller always has it in hand, e.g. the
    /// `ValueInterner`-bound decoded-id comparator `orBookkeeping` uses), so the
    /// store stays a pure container with no lifetime coupling to an interner.
    ///
    /// Unlike `CsrValueStore` (append-to-tail only), an interior key's run CAN
    /// grow — a sorted splice via `insertValueAt`, an O(value-tail + key-tail)
    /// shift — and a whole key's run CAN be dropped (the owning
    /// `HashMap::eraseSetIf` run-aware compaction). The exposed primitives
    /// (`runStartRaw` / `setRunStartRaw`, `valueRaw` / `setValueRaw`,
    /// `insertValueAt` / `eraseValueAt`, `truncateRunStarts` / `truncateValues`)
    /// are the 0-based raw-column building blocks those owning-map operations
    /// splice with. Two deload tags (the run-start column + the value column),
    /// the same shape `CsrValueStore` deloads — so the bytes are a pure function
    /// of logical content (I-103).
    ///
    /// @see `HashMap`, `CsrValueStore`, `ColdSetMap`,
    ///      D-168, I-118.
    template <typename V>
    class SetValueStore {
    public:
        /// @brief The stored value type (the owning `HashMap`'s value type).
        using ValueType = V;

        /// @brief Deload tags this value side contributes: run-starts + values.
        static constexpr int kTagCount = 2;

        /// @brief Bind the run-start index + value column to the LB arena +
        ///        dirty flag.
        ///
        /// @param arena The LB's bump arena; outlives the store.
        /// @param dirty The aggregate's shared content-change state.
        SetValueStore(LbArena* arena, DirtyState* dirty)
            : runStarts_(arena, dirty), values_(arena, dirty) {}

        SetValueStore(const SetValueStore&) = delete;
        SetValueStore& operator=(const SetValueStore&) = delete;

        /// @brief Uniform contract: clear the run-start + value columns.
        void clearValues() { runStarts_.clear(); values_.clear(); }

        /// @brief Uniform contract: release the run-start + value columns.
        void releaseValues() { runStarts_.release(); values_.release(); }

        /// @brief Uniform contract: deep-copy both columns (cross-arena).
        ///
        /// @param other Source store; resident.
        void copyValuesFrom(const SetValueStore& other) {
            runStarts_ = other.runStarts_;
            values_ = other.values_;
        }

        /// @brief Uniform contract: value-side live bytes (runs + values).
        ///
        /// @return Live bytes the runs index and values occupy.
        int64_t valuesLiveBytes() const {
            return runStarts_.liveBytes() + values_.liveBytes();
        }

        /// @brief Open a fresh empty run for a brand-new key at the current
        ///        value end — the new-key half of `insertSorted` / `assignSet`.
        void openRun() { runStarts_.push_back(values_.size()); }

        /// @brief Append one value at the current value end — the new-key fill
        ///        of `assignSet` (the caller supplies values already sorted).
        ///
        /// @param v The value to append.
        void appendValue(const V& v) { values_.push_back(v); }

        /// @brief Total number of values across every run.
        ///
        /// @return Value count.
        int32_t valueCount() const { return values_.size(); }

        /// @brief Number of values in key `id`'s run — DERIVED from the CSR
        ///        offsets (next key's start, or the value total for the last
        ///        key).
        ///
        /// @param id       A stored id; `1 <= id <= keyCount`.
        /// @param keyCount The owning map's key count (to bound the last run).
        /// @return The run length.
        int32_t runLen(int32_t id, int32_t keyCount) const {
            const int32_t start = runStarts_[id - 1];
            const int32_t end = (id < keyCount) ? runStarts_[id]
                                                : values_.size();
            return end - start;
        }

        /// @brief Value `j` of key `id`'s run.
        ///
        /// @param id A stored id.
        /// @param j  Position within the run (`0 <= j < runLen(id)`).
        /// @return Const reference to the value.
        const V& valueAt(int32_t id, int32_t j) const {
            return values_[runStarts_[id - 1] + j];
        }

        /// @brief Run-start offset of the key at 0-based position `i` — the
        ///        splice primitives' raw read into the run-start column.
        ///
        /// @param i Position in `[0, count())`.
        /// @return The value-column offset where that key's run begins.
        int32_t runStartRaw(int32_t i) const { return runStarts_[i]; }

        /// @brief In-place overwrite of run-start position `i` — the splice's
        ///        run-offset adjust (after an interior insert / erase / compaction).
        ///
        /// @param i Position in `[0, count())`.
        /// @param v The new run-start offset.
        void setRunStartRaw(int32_t i, int32_t v) { runStarts_.setAt(i, v); }

        /// @brief Raw value at 0-based column position `pos` — the binary-search
        ///        / compaction read.
        ///
        /// @param pos Position in `[0, valueCount())`.
        /// @return Const reference to the value.
        const V& valueRaw(int32_t pos) const { return values_[pos]; }

        /// @brief In-place overwrite of value at column position `pos` — the
        ///        compaction's "slide a survivor value down" + the same-size
        ///        `assignSet` overwrite.
        ///
        /// @param pos Position in `[0, valueCount())`.
        /// @param v   The new value.
        void setValueRaw(int32_t pos, const V& v) { values_.setAt(pos, v); }

        /// @brief Splice one value into the value column at `pos`, shifting the
        ///        tail right — the sorted-insert primitive.
        ///
        /// @details
        /// The owning `HashMap::insertSorted` follows it by bumping the
        /// run-starts of every later key by one. O(value-tail) via
        /// `PagedVector::insertAt`.
        ///
        /// @param pos Insert position in `[0, valueCount()]`.
        /// @param v   The value to splice in.
        void insertValueAt(int32_t pos, const V& v) { values_.insertAt(pos, v); }

        /// @brief Remove the value at column position `pos`, shifting the tail
        ///        left — the shrink half of `assignSet`.
        ///
        /// @param pos Position in `[0, valueCount())`.
        void eraseValueAt(int32_t pos) { values_.erase(pos); }

        /// @brief Drop the run-start column's tail to `n` keys — the compacting
        ///        erase's run-index truncate step.
        ///
        /// @param n Retained key count, in `[0, count()]`.
        void truncateRunStarts(int32_t n) { runStarts_.truncate(n); }

        /// @brief Drop the value column's tail to `n` values — the compacting
        ///        erase's value truncate step.
        ///
        /// @param n Retained value count, in `[0, valueCount()]`.
        void truncateValues(int32_t n) { values_.truncate(n); }

        /// @brief Dump the dense run-start column from `fromRow` — the runs tag.
        ///
        /// @param out     Byte sink.
        /// @param fromRow First key index; `0 <= fromRow <= count()`.
        void appendRunStartBytes(std::vector<char>& out, int32_t fromRow) const {
            runStarts_.appendSpanBytes(out, fromRow);
        }

        /// @brief Bulk-append `rowCount` run-start offsets — the runs reload
        ///        path.
        ///
        /// @param bytes    Source stream of `rowCount * 4` bytes.
        /// @param rowCount Run starts to append; >= 0.
        void bulkLoadRunStartBytes(const char* bytes, int64_t rowCount) {
            runStarts_.bulkAppendBytes(bytes, rowCount);
        }

        /// @brief Dump the dense value column from `fromValue` — the values tag.
        ///
        /// @param out       Byte sink.
        /// @param fromValue First value index; `0 <= fromValue <= valueCount()`.
        void appendValueBytes(std::vector<char>& out, int32_t fromValue) const {
            values_.appendSpanBytes(out, fromValue);
        }

        /// @brief Bulk-append `rowCount` values — the value-column reload path.
        ///
        /// @param bytes    Source stream of `rowCount * sizeof(V)` bytes.
        /// @param rowCount Values to append; >= 0.
        void bulkLoadValueBytes(const char* bytes, int64_t rowCount) {
            values_.bulkAppendBytes(bytes, rowCount);
        }

    private:
        PagedVector<int32_t> runStarts_;   // id-1 -> value-start offset (CSR)
        PagedVector<V> values_;            // all runs concatenated, key-id order
    };

    /// @brief Value-store policy for a key → an ordered RUN of VARIABLE-LENGTH
    ///        byte BLOBS, each blob one record's canonical serialization — the
    ///        "record value store" the cold-map family lacked.
    ///
    /// @details
    /// The shape no existing value store could hold: a value that is a multi-field
    /// record with variable-length fields (`EquivalenceClass`, and the Batch-4
    /// `HashMemory` record-set values). It stores the record as OPAQUE BYTES — a
    /// per-record codec at the call site serializes to / deserializes from a blob,
    /// so the store is record-agnostic and reused unchanged across record types.
    ///
    /// A **two-level CSR**, three cold paged columns, every boundary DERIVED (no
    /// redundant length column):
    ///   - `runStarts_`  — one int32 per key: the first blob index of key `id`'s
    ///     run (run length = next key's start, or the blob total for the last).
    ///   - `blobStarts_` — one int32 per blob: the byte offset where the blob
    ///     begins in the pool (blob length = next blob's start, or the pool total
    ///     for the last).
    ///   - `blobPool_`   — the blob bytes concatenated, DENSE (no no-straddle
    ///     padding: a blob is never handed out as a page-contiguous view, only
    ///     materialized through a codec, so it may straddle pages and reads walk
    ///     `contiguousRun` page-span by page-span into a heap buffer).
    /// All three are dense `PagedVector`s, so each deloads via `appendSpanBytes` /
    /// `bulkAppendBytes` directly (no lengths/bytes staging handshake — three
    /// independent dense tags, `kTagCount = 3`); the byte stream is a pure
    /// function of logical content (I-103), the codec
    /// being the single determinism point that canonicalizes every record field.
    ///
    /// The whole-run replace (`replaceRun`, the new-key path is `openRun` +
    /// `appendBlob`) and the run-aware compaction (the owning `HashMap::assignRun`
    /// / `eraseBlobIf`) splice the byte + blob columns once each via
    /// `PagedVector::replaceRange` / `moveBytes` — O(tail), the variable-length
    /// analog of `SetValueStore`'s `assignSet` / `eraseSetIf`. NO `ValueType`: the
    /// value is bytes, so the single-value / set surfaces never instantiate here
    /// (naming them on a `ColdBlobMap` is a compile error, the intended "use the
    /// blob surface" signal).
    ///
    /// @see `HashMap`, `ColdBlobMap`, `BytesKeyStore` (the byte-pool twin on the
    ///      key side), D-169, I-98.
    class BlobCsrValueStore {
    public:
        /// @brief Deload tags this value side contributes: run-starts +
        ///        blob-starts + blob-pool.
        static constexpr int kTagCount = 3;

        /// @brief Bind the three columns to the LB arena + dirty flag.
        ///
        /// @param arena The LB's arena; outlives the store.
        /// @param dirty The aggregate's shared content-change state.
        BlobCsrValueStore(LbArena* arena, DirtyState* dirty)
            : runStarts_(arena, dirty), blobStarts_(arena, dirty),
              blobPool_(arena, dirty) {}

        BlobCsrValueStore(const BlobCsrValueStore&) = delete;
        BlobCsrValueStore& operator=(const BlobCsrValueStore&) = delete;

        /// @brief Uniform contract: clear all three columns.
        void clearValues() {
            runStarts_.clear(); blobStarts_.clear(); blobPool_.clear();
        }

        /// @brief Uniform contract: release all three columns.
        void releaseValues() {
            runStarts_.release(); blobStarts_.release(); blobPool_.release();
        }

        /// @brief Uniform contract: deep-copy all three columns (cross-arena).
        ///
        /// @param other Source store; resident.
        void copyValuesFrom(const BlobCsrValueStore& other) {
            runStarts_ = other.runStarts_;
            blobStarts_ = other.blobStarts_;
            blobPool_ = other.blobPool_;
        }

        /// @brief Uniform contract: value-side live bytes (all three columns).
        ///
        /// @return Live bytes the three columns occupy.
        int64_t valuesLiveBytes() const {
            return runStarts_.liveBytes() + blobStarts_.liveBytes()
                 + blobPool_.liveBytes();
        }

        /// @brief Total blobs across every run.
        ///
        /// @return Blob count.
        int32_t blobCount() const { return blobStarts_.size(); }

        /// @brief Total blob-pool bytes — the blob-pool tag's element count.
        ///
        /// @return Pool byte count.
        int32_t poolByteCount() const { return blobPool_.size(); }

        /// @brief Number of blobs in key `id`'s run — DERIVED from the run-start
        ///        offsets (next key's start, or the blob total for the last key).
        ///
        /// @param id       A stored id; `1 <= id <= keyCount`.
        /// @param keyCount The owning map's key count (to bound the last run).
        /// @return The run length in blobs.
        int32_t runLen(int32_t id, int32_t keyCount) const {
            const int32_t start = runStarts_[id - 1];
            const int32_t end = (id < keyCount) ? runStarts_[id]
                                                : blobStarts_.size();
            return end - start;
        }

        /// @brief Byte offset where blob index `b` ends — the next blob's
        ///        start, or the pool end for the last blob.
        ///
        /// @param b Blob index in `[0, blobCount())`.
        /// @return The pool byte offset one past the blob.
        int32_t blobEnd(int32_t b) const {
            return (b + 1 < blobStarts_.size()) ? blobStarts_[b + 1] : blobPool_.size();
        }

        /// @brief Run-start (first blob index) of the key at 0-based position
        ///        `i` — the splice / compaction raw read.
        ///
        /// @param i Position in `[0, count())`.
        /// @return The blob index where that key's run begins.
        int32_t runStartRaw(int32_t i) const { return runStarts_[i]; }

        /// @brief In-place overwrite of run-start position `i` — the splice's
        ///        per-key offset adjust.
        ///
        /// @param i Position in `[0, count())`.
        /// @param v The new first-blob index.
        void setRunStartRaw(int32_t i, int32_t v) { runStarts_.setAt(i, v); }

        /// @brief Byte offset where blob index `b` begins (the pool end for
        ///        `b == blobCount()`, an empty run at the tail).
        ///
        /// @param b Blob index in `[0, blobCount()]`.
        /// @return The pool byte offset.
        int32_t runByteStart(int32_t b) const {
            return (b < blobStarts_.size()) ? blobStarts_[b] : blobPool_.size();
        }

        /// @brief In-place overwrite of blob-start position `b` — the
        ///        compaction's blob-offset rebase.
        ///
        /// @param b Position in `[0, blobCount())`.
        /// @param v The new byte offset.
        void setBlobStartRaw(int32_t b, int32_t v) { blobStarts_.setAt(b, v); }

        /// @brief Copy blob index `b`'s bytes into `out` (page-aware, sized to
        ///        the blob) — the read boundary the codec deserializes from.
        ///
        /// @param b   Blob index in `[0, blobCount())`.
        /// @param out Destination buffer (resized to the blob length).
        void readBlob(int32_t b, std::vector<char>& out) const {
            const int32_t start = blobStarts_[b];
            const int32_t end = blobEnd(b);
            const int32_t len = end - start;
            out.resize(static_cast<size_t>(len));
            int32_t pos = start, off = 0;
            while (pos < end) {
                int32_t run = 0;
                const char* p = blobPool_.contiguousRun(pos, run);
                if (run > end - pos) run = end - pos;
                std::memcpy(out.data() + off, p, static_cast<size_t>(run));
                pos += run; off += run;
            }
        }

        /// @brief Copy blob index `b`'s bytes into the caller-provided buffer
        ///        `out` (page-aware) — the no-heap twin of the
        ///        `std::vector<char>` `readBlob`, behind the arena-backed peek.
        ///
        /// @details
        /// The identical page-walking assembly as the `std::vector<char>`
        /// overload, minus the `resize`: the caller reserves exactly the blob's
        /// byte length up front (from the preceding `peekBlob`, which sets `len`
        /// before its contiguity test) and passes a pointer to at least that
        /// many writable bytes. `out` is filled with the same bytes the vector
        /// overload would write, in one contiguous run. Read side of the pool
        /// only; burst-safe.
        ///
        /// @param b   Blob index in `[0, blobCount())`.
        /// @param out Destination buffer, at least the blob's byte length.
        void readBlob(int32_t b, char* out) const {
            const int32_t start = blobStarts_[b];
            const int32_t end = blobEnd(b);
            int32_t pos = start, off = 0;
            while (pos < end) {
                int32_t run = 0;
                const char* p = blobPool_.contiguousRun(pos, run);
                if (run > end - pos) run = end - pos;
                std::memcpy(out + off, p, static_cast<size_t>(run));
                pos += run; off += run;
            }
        }

        /// @brief Zero-copy contiguous view of blob index `b`'s bytes WHEN the
        ///        blob lies on a single page — the no-allocation read primitive
        ///        behind the request-generation owner-set prune.
        ///
        /// @details
        /// Resolves the blob's `[start, end)` pool range and asks the pool for the
        /// contiguous run at `start` (`PagedVector::contiguousRun` — one shift +
        /// one mask + one resolve). When that run reaches the whole blob length the
        /// returned pointer spans every byte and the caller reads fields straight
        /// off the arena with no heap copy (the common case for the small
        /// owner-set records). When the blob straddles a page boundary the run
        /// falls short; this returns `false` and the caller falls back to the
        /// copying `readBlob`. An empty blob (`len == 0`) is trivially contiguous
        /// with a null pointer. Read-only — burst-safe on the shared LB the split
        /// executors read in parallel ([I-83](../30_invariants.md#i-83)).
        ///
        /// @param b   Blob index in `[0, blobCount())`.
        /// @param p   [out] Pointer to the blob's first byte, valid for `len`
        ///            bytes, when the result is `true`; `nullptr` on an empty blob;
        ///            unspecified when the result is `false`.
        /// @param len [out] The blob's byte length (always set).
        /// @return `true` when `p` spans all `len` bytes contiguously; `false` on a
        ///         page straddle.
        bool peekBlob(int32_t b, const char*& p, int32_t& len) const {
            const int32_t start = blobStarts_[b];
            const int32_t end = blobEnd(b);
            len = end - start;
            if (len == 0) { p = nullptr; return true; }
            int32_t run = 0;
            p = blobPool_.contiguousRun(start, run);
            return run >= len;
        }

        /// @brief Visit a consecutive blob range in order while caching the
        ///        current blob-index and byte-pool page spans.
        ///
        /// @details
        /// The sequential-scan twin of repeated @ref peekBlob calls. A scan loads
        /// each `blobStarts_` page once and reuses one resolved `blobPool_` page
        /// for every small blob it contains. A blob that crosses a byte-pool page
        /// is assembled once on @p scratch, exactly like the arena-backed
        /// `HashMap::peekBlobContiguous` fallback. The callback runs once per blob
        /// in logical order and may retain scratch-backed spans until its caller
        /// rewinds the arena; direct pool spans obey the normal resident-LB
        /// lifetime.
        ///
        /// Read-only and burst-safe (I-83). This is a scan operation, not a
        /// materialization: no descriptor list is built.
        ///
        /// @tparam Fn Callable `void(const char* bytes, int32_t len)`.
        /// @param b0      First blob index in `[0, blobCount()]`.
        /// @param count   Number of consecutive blobs; `b0 + count <= blobCount()`.
        /// @param scratch Arena used only for page-straddling blob copies.
        /// @param fn      Consumer invoked once per blob in order.
        /// @return Nothing.
        /// @invariant Visited bytes and lengths equal repeated `readBlob` results
        ///            for the same blob indices.
        /// @see peekBlob, readBlob.
        template <typename Fn>
        void forEachBlobRange(int32_t b0, int32_t count,
                              ScratchArena& scratch, Fn fn) const {
            assert(b0 >= 0 && count >= 0
                && b0 + count <= blobStarts_.size());
            const int32_t endBlob = b0 + count;
            const int32_t* startsPage = nullptr;
            int32_t startsBase = 0, startsCount = 0;
            const char* bytesPage = nullptr;
            int32_t bytesBase = 0, bytesCount = 0;
            const auto startAt = [&](int32_t b) -> int32_t {
                if (b == blobStarts_.size()) return blobPool_.size();
                if (startsPage == nullptr || b < startsBase
                    || b >= startsBase + startsCount) {
                    startsPage = blobStarts_.contiguousRun(b, startsCount);
                    startsBase = b;
                }
                return startsPage[b - startsBase];
            };
            for (int32_t b = b0; b < endBlob; ++b) {
                const int32_t start = startAt(b);
                const int32_t end = startAt(b + 1);
                const int32_t len = end - start;
                assert(len >= 0);
                if (len == 0) {
                    fn(nullptr, 0);
                    continue;
                }
                if (bytesPage != nullptr && start >= bytesBase
                    && end <= bytesBase + bytesCount) {
                    fn(bytesPage + (start - bytesBase), len);
                    continue;
                }
                bytesPage = blobPool_.contiguousRun(start, bytesCount);
                bytesBase = start;
                if (bytesCount >= len) {
                    fn(bytesPage, len);
                    continue;
                }
                char* copy = scratch.resolve(scratch.alloc(len, 1));
                int32_t pos = start, written = 0;
                while (pos < end) {
                    bytesPage = blobPool_.contiguousRun(pos, bytesCount);
                    bytesBase = pos;
                    const int32_t chunk = std::min(bytesCount, end - pos);
                    std::memcpy(copy + written, bytesPage,
                                static_cast<std::size_t>(chunk));
                    pos += chunk;
                    written += chunk;
                }
                assert(written == len
                    && "BlobCsrValueStore::forEachBlobRange copy diverged");
                fn(copy, len);
            }
        }

        /// @brief Open a fresh empty run for a brand-new key at the current blob
        ///        end — the new-key half of `HashMap::assignRun`.
        void openRun() { runStarts_.push_back(blobStarts_.size()); }

        /// @brief Append one blob at the current pool end — the new-key fill of
        ///        `HashMap::assignRun`.
        ///
        /// @param p Blob bytes; read only when `n > 0`.
        /// @param n Blob length; >= 0.
        void appendBlob(const char* p, int32_t n) {
            blobStarts_.push_back(blobPool_.size());
            if (n > 0) blobPool_.appendRun(p, n);
        }

        /// @brief Replace key id's blob run `[b0, b0+oldCount)` with `M` blobs —
        ///        the existing-key splice of `HashMap::assignRun`.
        ///
        /// @details
        /// One `replaceRange` on the pool (the run's bytes) and one on the
        /// blob-start column (the run's per-blob offsets), then a single O(blob-
        /// tail) rebase of every later blob start by the byte delta. The owning
        /// `HashMap::assignRun` follows with the O(key-tail) run-start rebase. The
        /// new blobs' absolute starts are computed from the run's byte origin, so
        /// the two-level CSR stays consistent. Single-threaded write side only
        /// (I-83).
        ///
        /// @param b0       First blob index of the key's run.
        /// @param oldCount Blobs currently in the run.
        /// @param bytes    The `M` new blobs' bytes concatenated; read only when
        ///                 the total is positive.
        /// @param lens     The `M` new blobs' per-blob lengths.
        /// @param M        New blob count; >= 0.
        void replaceRun(int32_t b0, int32_t oldCount, const char* bytes,
                        const int32_t* lens, int32_t M) {
            int32_t newBytes = 0;
            for (int32_t j = 0; j < M; ++j) newBytes += lens[j];
            replaceRunGenerated(b0, oldCount, M, newBytes,
                [&](const auto& sink) {
                    int32_t offset = 0;
                    for (int32_t j = 0; j < M; ++j) {
                        sink(bytes + offset, lens[j]);
                        offset += lens[j];
                    }
                });
        }

        /// @brief Replace one blob run from a replayable record emitter while
        ///        moving the byte and blob-index tails exactly once each.
        ///
        /// @details
        /// The segmented-source widening door for runs whose concatenated bytes
        /// cannot fit one arena block. @p emit is invoked twice: first to stream
        /// record byte spans directly into `blobPool_` through
        /// `PagedVector::replaceRangeGenerated`, then to generate the absolute
        /// `blobStarts_` prefix offsets from the same record lengths. Both dense
        /// CSR columns therefore retain exactly the bytes and boundaries that
        /// @ref replaceRun would produce, without a contiguous concat or heap
        /// prefix-start vector. Each surviving tail moves once.
        ///
        /// Any source span backed by this store's current pool must be preserved
        /// elsewhere before entry because the first generated replacement mutates
        /// that pool. Single-threaded write side only (I-83).
        ///
        /// @tparam Emit Replayable callable accepting a record sink with signature
        ///              `void(const char* bytes, int32_t len)`.
        /// @param b0       First blob index of the key's current run.
        /// @param oldCount Blobs currently in the run.
        /// @param M        Replacement blob count; >= 0.
        /// @param newBytes Total replacement byte count; >= 0.
        /// @param emit     Producer of exactly @p M records totalling
        ///                 @p newBytes bytes, in final run order.
        /// @return Nothing.
        /// @invariant The dense two-level CSR remains canonical: every generated
        ///            blob start is the prefix sum of the emitted lengths.
        /// @see replaceRun, PagedVector::replaceRangeGenerated.
        template <typename Emit>
        void replaceRunGenerated(int32_t b0, int32_t oldCount, int32_t M,
                                 int32_t newBytes, Emit emit) {
            assert(M >= 0 && newBytes >= 0);
            const int32_t y0 = runByteStart(b0);
            const int32_t oldBytes = runByteStart(b0 + oldCount) - y0;
            {
            RT_SCOPE_HERE("CSR_POOL_SHIFT_WRITE");
            if (newBytes != oldBytes)   // iter = pool bytes memmoved behind the run
                RT_NOTE_ITERATIONS_HERE(blobPool_.size() - (y0 + oldBytes));
            blobPool_.replaceRangeGenerated(y0, oldBytes, newBytes,
                [&](const auto& byteSink) {
                    int32_t emitted = 0, bytesEmitted = 0;
                    emit([&](const char* bytes, int32_t len) {
                        assert(len >= 0 && (len == 0 || bytes != nullptr));
                        byteSink(bytes, len);
                        ++emitted;
                        bytesEmitted += len;
                    });
                    assert(emitted == M && bytesEmitted == newBytes
                        && "BlobCsrValueStore::replaceRunGenerated byte emitter "
                           "shape mismatch");
                });
            } // RT_SCOPE CSR_POOL_SHIFT_WRITE
            int32_t acc = y0;
            {
            RT_SCOPE_HERE("CSR_BLOBSTARTS_SHIFT");
            if (M != oldCount || newBytes != oldBytes)   // iter = start entries touched behind the run
                RT_NOTE_ITERATIONS_HERE(blobStarts_.size() - (b0 + oldCount));
            blobStarts_.replaceRangeGenerated(b0, oldCount, M,
                [&](const auto& startSink) {
                    int32_t emitted = 0;
                    emit([&](const char*, int32_t len) {
                        startSink(&acc, 1);
                        acc += len;
                        ++emitted;
                    });
                    assert(emitted == M && acc == y0 + newBytes
                        && "BlobCsrValueStore::replaceRunGenerated start emitter "
                           "shape mismatch");
                });
            const int32_t byteDelta = newBytes - oldBytes;
            if (byteDelta != 0)
                blobStarts_.addScalarToSuffix(b0 + M, byteDelta);
            } // RT_SCOPE CSR_BLOBSTARTS_SHIFT
        }

        /// @brief The per-key record sink of @ref rebuildRunsGenerated: the
        ///        emitter produces a key's final run either record by record
        ///        (`blob`) or, for a key whose stored run is unchanged, as one
        ///        wholesale copy (`unchanged`).
        ///
        /// @details
        /// `unchanged(id)` marks the stored run of key `id` as carried over
        /// verbatim; CONSECUTIVE unchanged keys coalesce into one pending
        /// range that materializes as three chunked copies (the run starts
        /// re-based by one blob-index delta, the blob starts by one byte
        /// delta, the pool bytes as they are) when a produced key or the end
        /// of the rebuild follows — an untouched stretch of the map costs a
        /// memcpy of its bytes, not a call per key or per record. `blob`
        /// appends one record to the current key's run (materializing any
        /// pending range first, so the columns stay in id order). A key uses
        /// one of the two forms (asserted); a key that emits nothing gets an
        /// empty run. The sink is valid only inside the rebuild it belongs to.
        class RebuildSink {
        public:
            /// @brief Bind the sink to the store being rebuilt and the three
            ///        new columns; the emitter never constructs one.
            RebuildSink(BlobCsrValueStore& store, PagedVector<int32_t>& newRunStarts,
                        PagedVector<int32_t>& newBlobStarts,
                        PagedVector<char>& newPool, int32_t storedKeys)
                : store_(store), newRunStarts_(newRunStarts),
                  newBlobStarts_(newBlobStarts), newPool_(newPool),
                  storedKeys_(storedKeys) {}

            /// @brief Append one record to the current key's run.
            /// @param bytes The record bytes; read only when `len > 0`.
            /// @param len   The record length; `>= 0`.
            void blob(const char* bytes, int32_t len) {
                assert(len >= 0 && (len == 0 || bytes != nullptr)
                    && "RebuildSink::blob: record length without bytes");
                assert(!curUnchanged_
                    && "RebuildSink::blob: a key is either copied wholesale or emitted");
                if (!curStarted_) {
                    materializePending();
                    newRunStarts_.push_back(newBlobStarts_.size());
                    curStarted_ = true;
                }
                newBlobStarts_.push_back(newPool_.size());
                if (len > 0) newPool_.appendRun(bytes, len);
            }

            /// @brief Carry the stored run of key @p id over verbatim as the
            ///        current key's run (the key's records are unchanged).
            /// @param id The current key; a key with a stored run,
            ///           `1 <= id <= storedKeys`.
            void unchanged(int32_t id) {
                assert(id == curId_ && "RebuildSink::unchanged: not the current key");
                assert(id >= 1 && id <= storedKeys_
                    && "RebuildSink::unchanged: the key has no stored run");
                assert(!curStarted_ && !curUnchanged_
                    && "RebuildSink::unchanged: a key is either copied wholesale or emitted");
                if (pendingN_ > 0 && id == pendingFirst_ + pendingN_) {
                    ++pendingN_;
                } else {
                    materializePending();
                    pendingFirst_ = id;
                    pendingN_ = 1;
                }
                curUnchanged_ = true;
            }

            /// @brief The driver's per-key entry.
            /// @param id The key about to be emitted.
            void beginKey(int32_t id) {
                curId_ = id;
                curStarted_ = false;
                curUnchanged_ = false;
            }

            /// @brief The driver's per-key exit: a key that emitted nothing and
            ///        was not carried over gets an empty run.
            void endKey() {
                if (!curStarted_ && !curUnchanged_) {
                    materializePending();
                    newRunStarts_.push_back(newBlobStarts_.size());
                }
            }

            /// @brief The driver's final call: materialize a trailing range.
            void finish() { materializePending(); }

            /// @brief This sink writes (the scratch rebuild has one pass).
            bool isWritePass() const { return true; }

        private:
            /// @brief Copy the pending range of unchanged keys into the new
            ///        columns as three chunked copies.
            void materializePending() {
                if (pendingN_ == 0) return;
                const int32_t f = pendingFirst_;
                const int32_t l = pendingFirst_ + pendingN_ - 1;
                const int32_t b0 = store_.runStarts_[f - 1];
                const int32_t bEnd = (l < storedKeys_) ? store_.runStarts_[l]
                                                       : store_.blobStarts_.size();
                const int32_t y0 = store_.runByteStart(b0);
                const int32_t yEnd = store_.runByteStart(bEnd);
                appendShifted(newRunStarts_, store_.runStarts_, f - 1, l,
                              newBlobStarts_.size() - b0);
                appendShifted(newBlobStarts_, store_.blobStarts_, b0, bEnd,
                              newPool_.size() - y0);
                for (int32_t y = y0; y < yEnd;) {
                    int32_t run = 0;
                    const char* p = store_.blobPool_.contiguousRun(y, run);
                    if (run > yEnd - y) run = yEnd - y;
                    newPool_.appendRun(p, run);
                    y += run;
                }
                pendingN_ = 0;
            }

            /// @brief Append `src[from, to)` plus @p delta to @p dst in
            ///        stack-buffered chunks.
            static void appendShifted(PagedVector<int32_t>& dst,
                                      const PagedVector<int32_t>& src,
                                      int32_t from, int32_t to, int32_t delta) {
                int32_t buf[256];
                for (int32_t i = from; i < to;) {
                    int32_t run = 0;
                    const int32_t* p = src.contiguousRun(i, run);
                    if (run > to - i) run = to - i;
                    for (int32_t done = 0; done < run;) {
                        const int32_t n = std::min(256, run - done);
                        for (int32_t k = 0; k < n; ++k) buf[k] = p[done + k] + delta;
                        dst.appendRun(buf, n);
                        done += n;
                    }
                    i += run;
                }
            }

            BlobCsrValueStore& store_;
            PagedVector<int32_t>& newRunStarts_;
            PagedVector<int32_t>& newBlobStarts_;
            PagedVector<char>& newPool_;
            int32_t storedKeys_;
            int32_t curId_ = 0;
            bool curStarted_ = false;
            bool curUnchanged_ = false;
            int32_t pendingFirst_ = 0;
            int32_t pendingN_ = 0;
        };

        /// @brief Rebuild EVERY run from a per-key record emitter in one pass —
        ///        the batch write that replaces many interior splices.
        ///
        /// @details
        /// The flush door of the rule-index staging
        /// (D-333). @p emitAll is called once per key
        /// id in `1..keyCount` with a @ref RebuildSink and produces that key's
        /// complete final run — record by record (`blob`) or, for a key whose
        /// stored run is unchanged, as one wholesale copy (`unchanged`); the
        /// emitter may read this store's CURRENT runs while it produces (the
        /// old columns stay intact until the last key is emitted). The new
        /// columns are built on @p scratch in their final absolute form (run
        /// starts, blob starts, the pool), then the three dense CSR columns
        /// are refilled from them in page-sized chunks. The result is
        /// byte-identical to `openRun` + `appendBlob` of the same records in
        /// the same order for a fresh store, i.e. to what the same final runs
        /// reach through any sequence of `assignRun` / `appendBlobToRun`
        /// splices — with no tail move at all. Cost: a memcpy of the old pool
        /// plus the produced records, once, instead of O(pool tail) per
        /// splice. Forces `Restructured` (the whole value side is rewritten).
        /// Single-threaded write side only (I-83).
        ///
        /// A key beyond the stored ones (its run has not been opened yet) is
        /// legal: @p keyCount may exceed the run-start column's current size;
        /// the emitter supplies its run with `blob` like any other and the
        /// owner mints the key afterwards.
        ///
        /// @tparam EmitAll Callable `void(int32_t id, RebuildSink& sink)`
        ///                 invoked once per key id in ascending order.
        /// @param keyCount The owning map's key count after the owner's pending
        ///                 mints; every id in `1..keyCount` receives a run.
        /// @param emitAll  The per-key record producer.
        /// @param scratch  Per-slot scratch arena for the three collection
        ///                 columns (page tier); released before return.
        /// @return Nothing.
        /// @invariant After return: `runStarts_.size() == keyCount`, the blob
        ///            starts are the prefix sums of the emitted lengths, the
        ///            pool holds exactly the emitted bytes in emission order.
        /// @see replaceRunGenerated (the single-run splice), RebuildSink,
        ///      HashMap::assignAllRunsGenerated.
        template <typename EmitAll>
        void rebuildRunsGenerated(int32_t keyCount, EmitAll emitAll,
                                  LbArena& scratch) {
            const int32_t storedKeys = runStarts_.size();
            assert(keyCount >= storedKeys
                && "BlobCsrValueStore::rebuildRunsGenerated: fewer keys than runs");
            DirtyState scratchDirty = DirtyState::Clean;
            PagedVector<int32_t> newRunStarts(&scratch, &scratchDirty);
            PagedVector<int32_t> newBlobStarts(&scratch, &scratchDirty);
            PagedVector<char> newPool(&scratch, &scratchDirty);
            RebuildSink sink(*this, newRunStarts, newBlobStarts, newPool, storedKeys);
            {
                RT_SCOPE_HERE("CSR_REBUILD_EMIT");   // iter = keys walked
                RT_NOTE_ITERATIONS_HERE(keyCount);
                for (int32_t id = 1; id <= keyCount; ++id) {
                    sink.beginKey(id);
                    emitAll(id, sink);
                    sink.endKey();
                }
                sink.finish();
            }
            {
                RT_SCOPE_HERE("CSR_REBUILD_REFILL");   // iter = pool bytes copied back
                RT_NOTE_ITERATIONS_HERE(newPool.size());
                refillFrom(runStarts_, newRunStarts);
                refillFrom(blobStarts_, newBlobStarts);
                refillFrom(blobPool_, newPool);
            }
            assert(runStarts_.size() == keyCount
                && blobStarts_.size() == newBlobStarts.size()
                && blobPool_.size() == newPool.size()
                && "BlobCsrValueStore::rebuildRunsGenerated: rebuilt columns diverge");
            newPool.clear();
            newBlobStarts.clear();
            newRunStarts.clear();
        }

        /// @brief The single-pass sink of @ref rebuildRunsInPlace: a produced
        ///        key's records are buffered on scratch (proportional to the
        ///        produced runs, never to the map), a carried-over key is only
        ///        measured.
        ///
        /// @details
        /// The same `blob` / `unchanged` protocol as @ref RebuildSink. The one
        /// emit pass runs BEFORE any column changes, so the emitter reads the
        /// store's intact runs; `isWritePass()` is `true` (an emitter's
        /// write-only side effect, the remaining-args reverse-index edge, runs
        /// in this pass).
        class ProduceSink {
        public:
            /// @brief Bind to the store being rebuilt and the produced-run scratch.
            ProduceSink(const BlobCsrValueStore& store, int32_t storedKeys,
                        PagedVector<int32_t>& prodLens, PagedVector<char>& prodPool)
                : store_(store), storedKeys_(storedKeys),
                  prodLens_(prodLens), prodPool_(prodPool) {}

            /// @brief Buffer one produced record.
            void blob(const char* bytes, int32_t len) {
                assert(len >= 0 && (len == 0 || bytes != nullptr)
                    && "ProduceSink::blob: record length without bytes");
                assert(!unchanged_
                    && "ProduceSink::blob: a key is either copied wholesale or emitted");
                prodLens_.push_back(len);
                if (len > 0) prodPool_.appendRun(bytes, len);
                ++blobs_;
                bytes_ += len;
            }

            /// @brief Measure the stored run of the current key as carried over.
            void unchanged(int32_t id) {
                assert(id == curId_ && "ProduceSink::unchanged: not the current key");
                assert(id >= 1 && id <= storedKeys_
                    && "ProduceSink::unchanged: the key has no stored run");
                assert(blobs_ == 0 && !unchanged_
                    && "ProduceSink::unchanged: a key is either copied wholesale or emitted");
                const int32_t b0 = store_.runStarts_[id - 1];
                const int32_t bEnd = (id < storedKeys_) ? store_.runStarts_[id]
                                                        : store_.blobStarts_.size();
                blobs_ = bEnd - b0;
                bytes_ = store_.runByteStart(bEnd) - store_.runByteStart(b0);
                unchanged_ = true;
            }

            /// @brief The one pass carries the side effects.
            bool isWritePass() const { return true; }

            /// @brief The driver's per-key entry.
            void beginKey(int32_t id) {
                curId_ = id;
                blobs_ = 0;
                bytes_ = 0;
                unchanged_ = false;
                lensStart_ = prodLens_.size();
                poolStart_ = prodPool_.size();
            }

            int32_t blobs() const { return blobs_; }          ///< The key's final blob count.
            int64_t bytes() const { return bytes_; }          ///< The key's final byte count.
            bool produced() const { return !unchanged_; }     ///< Produced (rewritten) key?
            int32_t lensStart() const { return lensStart_; }  ///< Its first record length slot.
            int32_t poolStart() const { return poolStart_; }  ///< Its first scratch pool byte.

        private:
            const BlobCsrValueStore& store_;
            int32_t storedKeys_;
            PagedVector<int32_t>& prodLens_;
            PagedVector<char>& prodPool_;
            int32_t curId_ = 0;
            int32_t blobs_ = 0;
            int64_t bytes_ = 0;
            bool unchanged_ = false;
            int32_t lensStart_ = 0;
            int32_t poolStart_ = 0;
        };

        /// @brief Rebuild every run IN PLACE from a per-key record emitter —
        ///        one emit pass, nothing before the first produced key moves,
        ///        scratch proportional to the produced runs only.
        ///
        /// @details
        /// The production flush door of the rule-index staging
        /// (D-333). Pass one calls @p emitAll once per
        /// key on the intact store (`ProduceSink`): a produced key's records
        /// are buffered on @p scratch, a carried-over key is measured. Pass
        /// two grows the three columns to the final sizes
        /// (`PagedVector::growTo`) and walks from the last key down to the
        /// FIRST produced key writing the tail backwards from the new end: a
        /// stretch of carried-over keys moves once as one range (`moveRange`
        /// on the pool, delta-shifted moves of the blob starts and run
        /// starts), a produced key's buffered run is placed. Keys before the
        /// first produced key are untouched, so the cost is O(bytes behind the
        /// first produced key) plus the produced records — the cost of ONE
        /// splice at that key — and the only memory growth is the delta. The
        /// walk is safe because a produced run never shrinks (asserted: every
        /// merge only adds), so every write lands at or above the old
        /// position of the data it replaces and the unprocessed keys' old
        /// data stays intact below. Result byte-identical to
        /// @ref rebuildRunsGenerated (the scratch twin, the test oracle).
        /// Forces `Restructured`. Single-threaded write side only (I-83).
        ///
        /// @tparam EmitAll Callable `void(int32_t id, ProduceSink& sink)`
        ///                 invoked once per key id in ascending order.
        /// @param keyCount The owning map's key count after the owner's pending
        ///                 mints; every id in `1..keyCount` receives a run.
        /// @param emitAll  The per-key record producer.
        /// @param scratch  Per-slot scratch arena for the per-key size tables
        ///                 and the produced-run buffer (page tier).
        /// @return Nothing.
        /// @invariant After return the columns equal those @ref
        ///            rebuildRunsGenerated produces for the same emitter.
        /// @see rebuildRunsGenerated, ProduceSink, HashMap::assignAllRunsGenerated.
        template <typename EmitAll>
        void rebuildRunsInPlace(int32_t keyCount, EmitAll emitAll, LbArena& scratch) {
            const int32_t storedKeys = runStarts_.size();
            assert(keyCount >= storedKeys
                && "BlobCsrValueStore::rebuildRunsInPlace: fewer keys than runs");
            DirtyState scratchDirty = DirtyState::Clean;
            PagedVector<int32_t> newBlobs(&scratch, &scratchDirty);
            PagedVector<int32_t> newBytes(&scratch, &scratchDirty);
            PagedVector<int32_t> lensStart(&scratch, &scratchDirty);
            PagedVector<int32_t> poolStart(&scratch, &scratchDirty);
            PagedVector<char> produced(&scratch, &scratchDirty);
            PagedVector<int32_t> prodLens(&scratch, &scratchDirty);
            PagedVector<char> prodPool(&scratch, &scratchDirty);
            int64_t totalBlobs = 0;
            int64_t totalBytes = 0;
            int32_t firstProduced = 0;
            {
                RT_SCOPE_HERE("CSR_REBUILD_EMIT");   // iter = keys emitted
                RT_NOTE_ITERATIONS_HERE(keyCount);
                ProduceSink ps(*this, storedKeys, prodLens, prodPool);
                for (int32_t id = 1; id <= keyCount; ++id) {
                    ps.beginKey(id);
                    emitAll(id, ps);
                    assert(ps.bytes() <= INT32_MAX
                        && "rebuildRunsInPlace: a run exceeds the pool's index range");
                    if (ps.produced() && id <= storedKeys) {
                        const int32_t b0 = runStarts_[id - 1];
                        const int32_t bEnd = (id < storedKeys) ? runStarts_[id]
                                                                : blobStarts_.size();
                        assert(ps.blobs() >= bEnd - b0
                            && ps.bytes() >= runByteStart(bEnd) - runByteStart(b0)
                            && "rebuildRunsInPlace: a produced run shrank - the merges only add");
                    }
                    assert((ps.produced() || id <= storedKeys)
                        && "rebuildRunsInPlace: a key without a stored run must be produced");
                    newBlobs.push_back(ps.blobs());
                    newBytes.push_back(static_cast<int32_t>(ps.bytes()));
                    lensStart.push_back(ps.lensStart());
                    poolStart.push_back(ps.poolStart());
                    produced.push_back(ps.produced() ? 1 : 0);
                    if (ps.produced() && firstProduced == 0) firstProduced = id;
                    totalBlobs += ps.blobs();
                    totalBytes += ps.bytes();
                }
            }
            assert(totalBlobs <= INT32_MAX && totalBytes <= INT32_MAX
                && "rebuildRunsInPlace: the rebuilt columns exceed the index range");
            if (firstProduced != 0) {
                const int32_t oldBytes = blobPool_.size();
                // Where the untouched prefix ends - the cursors must land exactly here.
                const int32_t prefixBlobs = (firstProduced <= storedKeys)
                    ? runStarts_[firstProduced - 1] : blobStarts_.size();
                const int32_t prefixBytes = runByteStart(prefixBlobs);
                RT_SCOPE_HERE("CSR_REBUILD_WRITE");   // iter = pool bytes behind the first produced key
                RT_NOTE_ITERATIONS_HERE(oldBytes - prefixBytes);
                runStarts_.growTo(keyCount);
                blobStarts_.growTo(static_cast<int32_t>(totalBlobs));
                blobPool_.growTo(static_cast<int32_t>(totalBytes));

                int32_t blobCursor = static_cast<int32_t>(totalBlobs);
                int32_t byteCursor = static_cast<int32_t>(totalBytes);
                int32_t pendingLo = 0, pendingHi = 0;       // a carried-over range [lo, hi]
                int32_t pendingBlobs = 0, pendingBytes = 0;
                const auto movePending = [&]() {
                    if (pendingHi == 0) return;
                    const int32_t b0 = runStarts_[pendingLo - 1];
                    const int32_t y0 = blobStarts_[b0];
                    const int32_t destBlob = blobCursor - pendingBlobs;
                    const int32_t destByte = byteCursor - pendingBytes;
                    assert(destBlob >= b0 && destByte >= y0
                        && "rebuildRunsInPlace: a carried-over range would move down");
                    blobPool_.moveRange(y0, destByte, pendingBytes);
                    moveShiftedDescending(blobStarts_, b0, destBlob, pendingBlobs, destByte - y0);
                    moveShiftedDescending(runStarts_, pendingLo - 1, pendingLo - 1,
                                          pendingHi - pendingLo + 1, destBlob - b0);
                    blobCursor = destBlob;
                    byteCursor = destByte;
                    pendingLo = pendingHi = 0;
                    pendingBlobs = pendingBytes = 0;
                };
                for (int32_t id = keyCount; id >= firstProduced; --id) {
                    if (produced[id - 1] == 0) {
                        if (pendingHi == 0) pendingHi = id;
                        pendingLo = id;
                        pendingBlobs += newBlobs[id - 1];
                        pendingBytes += newBytes[id - 1];
                        continue;
                    }
                    movePending();
                    const int32_t M = newBlobs[id - 1];
                    const int32_t B = newBytes[id - 1];
                    const int32_t ls = lensStart[id - 1];
                    const int32_t pst = poolStart[id - 1];
                    blobCursor -= M;
                    byteCursor -= B;
                    for (int32_t at = 0; at < B;) {
                        int32_t run = 0;
                        const char* p = prodPool.contiguousRun(pst + at, run);
                        if (run > B - at) run = B - at;
                        blobPool_.writeRunAt(byteCursor + at, p, run);
                        at += run;
                    }
                    {
                        int32_t buf[256];
                        int32_t acc = byteCursor;
                        for (int32_t j = 0; j < M;) {
                            const int32_t n = std::min(256, M - j);
                            for (int32_t k = 0; k < n; ++k) {
                                buf[k] = acc;
                                acc += prodLens[ls + j + k];
                            }
                            blobStarts_.writeRunAt(blobCursor + j, buf, n);
                            j += n;
                        }
                        assert(acc == byteCursor + B
                            && "rebuildRunsInPlace: the produced run's lengths do not sum to its bytes");
                    }
                    runStarts_.setAt(id - 1, blobCursor);
                }
                movePending();
                assert(blobCursor == prefixBlobs && byteCursor == prefixBytes
                    && "rebuildRunsInPlace: the backward walk did not land on the untouched prefix");
            } else {
                // Every key carried over: the columns already hold the result.
                assert(keyCount == storedKeys);
            }
            prodPool.clear();
            prodLens.clear();
            produced.clear();
            poolStart.clear();
            lensStart.clear();
            newBytes.clear();
            newBlobs.clear();
        }

        /// @brief Move `src[from, from+count)` to `dest..` adding @p delta to
        ///        every element, walking from the high end so an overlapping
        ///        upward move never reads a slot it already wrote
        ///        (`dest >= from`).
        ///
        /// @param v     The column.
        /// @param from  First source slot.
        /// @param dest  First destination slot; `>= from`.
        /// @param count Elements; `<= 0` is a no-op.
        /// @param delta Added to every moved element.
        /// @return Nothing.
        static void moveShiftedDescending(PagedVector<int32_t>& v, int32_t from,
                                          int32_t dest, int32_t count, int32_t delta) {
            assert(dest >= from && "moveShiftedDescending: downward move");
            if (count <= 0) return;
            int32_t buf[256];
            for (int32_t rem = count; rem > 0;) {
                const int32_t n = std::min(256, rem);
                const int32_t s = from + rem - n;
                for (int32_t k = 0; k < n; ++k) buf[k] = v[s + k] + delta;
                v.writeRunAt(dest + rem - n, buf, n);
                rem -= n;
            }
        }

        /// @brief Replace a column's content with another column's, copied in
        ///        page-sized chunks (the rebuild's adoption step).
        ///
        /// @tparam T   The element type.
        /// @param dst  The column to refill (cleared first; `Restructured`).
        /// @param src  The source column (on any arena).
        /// @return Nothing.
        template <typename T>
        static void refillFrom(PagedVector<T>& dst, const PagedVector<T>& src) {
            dst.clear();
            for (int32_t at = 0; at < src.size();) {
                int32_t run = 0;
                const T* p = src.contiguousRun(at, run);
                dst.appendRun(p, run);
                at += run;
            }
        }

        /// @brief Rebase every key-run start from @p first by one blob-count
        ///        delta using the paged column's bulk suffix door.
        ///
        /// @details
        /// An interior blob-run replacement shifts every later key's first blob
        /// by the same count. The page-run update is byte-identical to repeated
        /// `runStartRaw` plus `setRunStartRaw` calls while avoiding two directory
        /// resolutions per integer.
        ///
        /// @param first First zero-based key-run-start slot to change.
        /// @param delta Blob-count delta added to every suffix slot.
        /// @return Nothing.
        /// @invariant Run-start order and differences remain unchanged; only the
        ///            shared absolute base of the suffix moves by @p delta.
        void addToRunStartsSuffix(int32_t first, int32_t delta) {
            runStarts_.addScalarToSuffix(first, delta);
        }

        /// @brief Move `n` pool bytes from `srcByte` down to `destByte`
        ///        (`destByte <= srcByte`) — the compaction's run-bytes slide.
        ///
        /// @details
        /// Forward copy (dest below source, non-overlapping in the survivor-
        /// compaction direction), page-aware via `contiguousRun` reads + `setAt`
        /// writes. Single-threaded write side only (I-83).
        ///
        /// @param destByte Destination pool offset.
        /// @param srcByte  Source pool offset; `>= destByte`.
        /// @param n        Byte count; >= 0.
        void moveBytes(int32_t destByte, int32_t srcByte, int32_t n) {
            assert(destByte <= srcByte);
            int32_t done = 0;
            while (done < n) {
                int32_t run = 0;
                const char* p = blobPool_.contiguousRun(srcByte + done, run);
                if (run > n - done) run = n - done;
                for (int32_t k = 0; k < run; ++k)
                    blobPool_.setAt(destByte + done + k, p[k]);
                done += run;
            }
        }

        /// @brief Drop the run-start column's tail to `n` keys — the compaction's
        ///        run-index truncate.
        ///
        /// @param n Retained key count, in `[0, count()]`.
        void truncateRunStarts(int32_t n) { runStarts_.truncate(n); }

        /// @brief Drop the blob-start column's tail to `n` blobs — the
        ///        compaction's blob-index truncate.
        ///
        /// @param n Retained blob count, in `[0, blobCount()]`.
        void truncateBlobStarts(int32_t n) { blobStarts_.truncate(n); }

        /// @brief Drop the blob pool's tail to `n` bytes — the compaction's pool
        ///        truncate.
        ///
        /// @param n Retained byte count, in `[0, poolByteCount()]`.
        void truncatePool(int32_t n) { blobPool_.truncate(n); }

        /// @brief Dump the dense run-start column from `fromRow` — the runs tag.
        ///
        /// @param out     Byte sink.
        /// @param fromRow First key index.
        void appendRunStartBytes(std::vector<char>& out, int32_t fromRow) const {
            runStarts_.appendSpanBytes(out, fromRow);
        }

        /// @brief Bulk-append run-start offsets — the runs reload path.
        ///
        /// @param bytes    Source stream of `rowCount * 4` bytes.
        /// @param rowCount Run starts to append; >= 0.
        void bulkLoadRunStartBytes(const char* bytes, int64_t rowCount) {
            runStarts_.bulkAppendBytes(bytes, rowCount);
        }

        /// @brief Dump the dense blob-start column from `fromRow` — the
        ///        blob-starts tag.
        ///
        /// @param out     Byte sink.
        /// @param fromRow First blob index.
        void appendBlobStartBytes(std::vector<char>& out, int32_t fromRow) const {
            blobStarts_.appendSpanBytes(out, fromRow);
        }

        /// @brief Bulk-append blob-start offsets — the blob-starts reload path.
        ///
        /// @param bytes    Source stream of `rowCount * 4` bytes.
        /// @param rowCount Blob starts to append; >= 0.
        void bulkLoadBlobStartBytes(const char* bytes, int64_t rowCount) {
            blobStarts_.bulkAppendBytes(bytes, rowCount);
        }

        /// @brief Dump the dense blob-pool bytes from byte offset `fromByte` —
        ///        the blob-pool tag.
        ///
        /// @param out      Byte sink.
        /// @param fromByte First pool byte offset.
        void appendBlobPoolBytes(std::vector<char>& out, int32_t fromByte) const {
            blobPool_.appendSpanBytes(out, fromByte);
        }

        /// @brief Bulk-append blob-pool bytes — the blob-pool reload path.
        ///
        /// @param bytes    Source stream of `rowCount` bytes.
        /// @param rowCount Bytes to append; >= 0.
        void bulkLoadBlobPoolBytes(const char* bytes, int64_t rowCount) {
            blobPool_.bulkAppendBytes(bytes, rowCount);
        }

    private:
        PagedVector<int32_t> runStarts_;   // id-1 -> first blob index (CSR/blobs)
        PagedVector<int32_t> blobStarts_;  // blob -> byte offset    (CSR/bytes)
        PagedVector<char> blobPool_;       // all blob bytes, dense, key-id order

    };

    /// @brief The one cold hash container — cold keys + a heap open-addressing
    ///        index + a value-store policy. Set / map / multimap are the three
    ///        `ValueStore` instantiations (`ColdHashSet` / `ColdHashMap` /
    ///        `ColdMultiMap` aliases).
    ///
    /// @details
    /// `HashMap` is the single class the cold-map family collapsed to. The engine
    /// — `mint` / `lookup` / `decode`, the throw-away `PagedHashIndex`
    /// (`rebuildIndex` / `indexInsert` / `indexPlace`), and the lifecycle
    /// (`copyFrom` / `resetToFresh` / `release`) — lives here ONCE, and the value
    /// column is a policy (the second template parameter), symmetric to the
    /// `KeyStore` policy (`BytesKeyStore` / `PodKeyStore<K>`):
    ///
    ///   - `EmptyValueStore` → a SET (`ColdHashSet`): keys only.
    ///   - `SingleValueStore<V>` → a single-value MAP (`ColdHashMap`).
    ///   - `CsrValueStore<V>` → a multi-value MAP (`ColdMultiMap`, CSR runs).
    ///
    /// Every value store exposes the same uniform contract (`clearValues` /
    /// `releaseValues` / `copyValuesFrom` / `valuesLiveBytes` + a `kTagCount`),
    /// so the always-instantiated whole-object methods call it unconditionally —
    /// no `if constexpr`. The value-shaped surface (`insert` / `find` /
    /// `valueAt` / `appendToTail` / `runLen` / `valueCount` + the value deload
    /// helpers) and the byte-key legacy surface (`intern` / `view` /
    /// `decodeString` + the `LengthsView` / `BytesView` facets) instantiate ONLY
    /// for the instantiation that calls them — the same on-demand member
    /// instantiation the legacy byte-key forwarders already relied on.
    ///
    /// The value store is a private empty-base (`: private ValueStore`) so a
    /// set's `EmptyValueStore` is elided (`[[no_unique_address]]` is ignored
    /// under the project's C++17/MSVC — empty-base optimization is the portable
    /// zero-cost form), keeping `ColdStringTable`'s layout byte-identical.
    ///
    /// key → id is the heap index (FNV content hash, linear probing, ids in
    /// int32 slots); id → key is the key store's positional `decodeAt`. The key
    /// columns and the value columns are cold paged streams; the index is
    /// DERIVED — rebuilt on reload, never deloaded (I-117).
    ///
    /// Determinism: interning the same key sequence (and appending the same
    /// values) yields the same ids, arena layout, and deload bytes; reload
    /// re-bumps element-by-element in id order, so every id reproduces exactly.
    ///
    /// Threading: `mint` / `insert` / `appendToTail` are single-threaded
    /// write-side only (I-83); `lookup` / `find` / `decode` are safe from the
    /// parallel burst (read-only, non-minting).
    ///
    /// @invariant Append-only: an id, once minted, resolves to the same key for
    ///            the container's lifetime (until `resetToFresh` / `release`).
    /// @invariant The index is a pure function of the stored keys — rebuilt,
    ///            never persisted; no observable depends on its layout
    ///            (I-117).
    /// @see `BytesKeyStore`, `PodKeyStore`, `EmptyValueStore`,
    ///      `SingleValueStore`, `CsrValueStore`, D-165.
    template <typename KeyStore, typename ValueStore>
    class HashMap : private ValueStore {
    public:
        /// @brief Probe type `mint` / `lookup` / `insert` accept (from the key
        ///        store).
        using KeyView = typename KeyStore::KeyView;

        /// @brief Decode result of `decode` / `keyAt` (from the key store).
        using KeyDecode = typename KeyStore::KeyDecode;

        /// @brief Number of deload tags this container contributes — the key
        ///        store's plus the value store's (`I-117`
        ///        keeps the index out of the count).
        static constexpr int kTagCount =
            KeyStore::kTagCount + ValueStore::kTagCount;

        /// @brief Bind to the owning LB's arena and the aggregate's dirty flag.
        ///
        /// @param arena The LB's bump arena; outlives the container.
        /// @param dirty The aggregate's shared content-change state.
        HashMap(LbArena* arena, DirtyState* dirty)
            : ValueStore(arena, dirty), arena_(arena), ks_(arena, dirty),
              buckets_(arena) {
            assert(arena != nullptr && dirty != nullptr);
        }

        HashMap(const HashMap&) = delete;
        HashMap& operator=(const HashMap&) = delete;
        HashMap(HashMap&&) = delete;
        HashMap& operator=(HashMap&&) = delete;

        /// @brief Number of keys minted (ids run 1..count()).
        ///
        /// @return Key count.
        int32_t count() const { return ks_.count(); }

        /// @brief Whether no key has been minted.
        ///
        /// @return `true` when `count() == 0`.
        bool empty() const { return ks_.empty(); }

        /// @brief Approximate live byte footprint (cold key columns + the
        ///        throw-away paged hash buckets — both ride the LB arena).
        ///
        /// @return Live bytes the keys and the hash index occupy.
        int64_t liveBytes() const {
            return ks_.liveBytes() + buckets_.liveBytes()
                 + ValueStore::valuesLiveBytes();
        }

        /// @brief Bytes held by the DERIVED key→id index alone.
        ///
        /// @details
        /// The `PagedHashIndex` slot array is rebuilt on reload and never
        /// deloaded ([I-117]), so it appears in no `ContainerTag` and the
        /// deload image understates the container's real RAM by exactly this
        /// much. Telemetry only — the memory measurement reads it to separate
        /// derived-index cost from block and page slack; nothing in the prover
        /// branches on it (Rule 16).
        ///
        /// @return Live bytes of the throw-away hash index.
        /// @see `indexBytes()` on the key facet views, `mem_tracker.hpp`.
        int64_t indexBytes() const { return buckets_.liveBytes(); }

        /// @brief The cold key store — the column-dump / reload escape hatch for
        ///        the embedding map forms (`ColdHashMap` / `ColdMultiMap`) and
        ///        the deload round-trip tests.
        ///
        /// @details
        /// Read access for dumping the key column(s); write access for the bulk
        /// reload path, which appends keys raw and MUST be followed by
        /// `rebuildIndex()` (raw `appendKey` bypasses the index by design — the
        /// reload rebuilds it once at the end).
        ///
        /// @return Reference to the owned key store.
        KeyStore& keyStore() { return ks_; }

        /// @brief Const overload of `keyStore` — for dumping the key column(s).
        ///
        /// @return Const reference to the owned key store.
        const KeyStore& keyStore() const { return ks_; }

        /// @brief Find-or-mint: the interner encode primitive.
        ///
        /// @details
        /// Single-threaded write side only (I-83). A hit returns the existing id
        /// without touching anything; a miss appends the key (cold) at the next
        /// id and inserts that id into the throw-away paged hash index (growing
        /// it via a rebuild at 1/2 load). The cold key append escalates the
        /// aggregate dirty flag; the throw-away index never touches a dirty flag.
        ///
        /// @param k Key to intern.
        /// @return The key's id; >= 1.
        int32_t mint(const KeyView& k) {
            assert(arena_->resident()
                && "mint on a deloaded LB - reload at a sanctioned touch "
                   "point first (I-111)");
            const int32_t existing = lookup(k);
            if (existing != 0) return existing;
            ks_.appendKey(k);
            const int32_t id = ks_.count();
            indexInsert(id);
            ++insertEpoch_;
            // Consistency: the just-minted key must resolve to its own id
            // through the index (a cheap per-mint round-trip tripwire).
            assert(lookup(k) == id
                && "cold-index desync: minted key not findable at its id");
            return id;
        }

        /// @brief Non-minting probe: the interner lookup primitive.
        ///
        /// @details
        /// Safe from the parallel burst (pure read): an FNV content hash, then
        /// linear probing of the throw-away paged hash buckets with a key
        /// comparison against the cold keys. Residency is asserted FIRST — a
        /// deloaded set reads empty, and answering "not found" there is a silent
        /// lie (the IncubatorPeano1 incident); a genuinely empty RESIDENT set
        /// still misses as a defined result.
        ///
        /// @param k Key to look up.
        /// @return The id, or 0 when the key was never minted.
        int32_t lookup(const KeyView& k) const {
            assert(arena_->resident()
                && "lookup on a deloaded LB - reload at a sanctioned touch "
                   "point first (I-111)");
            if (buckets_.empty()) return 0;
            const uint64_t mask = buckets_.capacity() - 1;
            const uint64_t probeHash = ks_.hashProbe(k);
            uint64_t i = probeHash & mask;
            while (true) {
                const int32_t id = buckets_.at(static_cast<int32_t>(i));
                if (id == 0) return 0;
                if (ks_.equalStored(id, k, probeHash)) return id;
                i = (i + 1) & mask;
            }
        }

        /// @brief Decode an id back to its key (the cold, positional direction).
        ///
        /// @param id A minted id; `1 <= id <= count()`.
        /// @return The key (a view for byte keys; a const ref for POD keys).
        KeyDecode decode(int32_t id) const { return ks_.decodeAt(id); }

        /// @brief Decode an id back to its key — the map / multimap spelling of
        ///        `decode`.
        ///
        /// @param id A minted id; `1 <= id <= count()`.
        /// @return The key (a view for byte keys; a const ref for POD keys).
        KeyDecode keyAt(int32_t id) const { return ks_.decodeAt(id); }

        /// @brief Whether `k` is present (any value-store shape).
        ///
        /// @param k The key to test.
        /// @return `true` when `k` has been minted.
        bool contains(const KeyView& k) const { return lookup(k) != 0; }

        // ---- Erase surface (POD-key set + single-value map) ----------------
        // Both call `KeyStore::eraseAt` + `ValueStore::eraseValueAt`, so they
        // instantiate only where both exist (POD-key set / map). A byte-key or
        // multimap erase is deliberately deferred — naming it there is a compile
        // error, the intended "not supported yet" signal.

        /// @brief Remove key `k` if present — the per-key erase.
        ///
        /// @details
        /// A miss is a defined query result (`false`), not a failure. Routed
        /// through the same compacting `eraseIf` pass via a one-key predicate —
        /// `erase` (the `eradicateImplicationFromLB` path) is rare, so the
        /// O(count()) sweep is fine and keeps a single erase mechanism.
        /// Single-threaded write side only (I-83).
        ///
        /// @param k The key to remove.
        /// @return `true` when a key was removed, `false` on a miss.
        bool erase(const KeyView& k) {
            return eraseIf([&k](const auto& kd) { return kd == k; }) != 0;
        }

        /// @brief Remove every key the predicate accepts — the bulk scope-wipe
        ///        primitive (the `Memory::wipeSubtree` driver).
        ///
        /// @details
        /// A single forward COMPACTION pass: survivors slide to the front of the
        /// key + value columns in lockstep (`moveKeyTo` / `moveValue` — the
        /// reviewed in-place writes), the dead tail is `truncate`d, and the
        /// throw-away index is rebuilt once. O(count()) with one rebuild — the
        /// "rebuild dense from live content" the substrate already runs at reload
        /// / reshuffle, scoped to one container. Survivor relative order is
        /// preserved, so the post-wipe id order — and the deload bytes — are a
        /// deterministic function of the surviving content (history invisible):
        /// byte-identical to a from-scratch insert of the survivors. Works for
        /// BOTH key stores (`moveKeyTo` is position-based — POD copies the key,
        /// the byte-key store slides only the location entry, its dead bytes
        /// reclaimed as holes by the copying compaction) with `EmptyValueStore`
        /// / `SingleValueStore`. A `ColdMultiMap` (`CsrValueStore`) omits the
        /// per-key `moveValue`, so naming `eraseIf` there is a compile error —
        /// the CSR set form erases through `eraseSetIf` (run-aware) instead.
        /// Single-threaded write side only (I-83).
        ///
        /// @tparam Pred A callable `bool(KeyDecode)` — erase the key when it
        ///         returns true.
        /// @param pred The erase predicate, tested against each decoded key.
        /// @return Number of keys removed.
        template <typename Pred>
        int32_t eraseIf(Pred pred) {
            assert(arena_->resident()
                && "eraseIf on a deloaded LB (I-111)");
            const int32_t n = ks_.count();
            int32_t write = 0;   // survivors compacted so far (0-based)
            for (int32_t read = 0; read < n; ++read) {
                const int32_t id = read + 1;        // 1-based id of this slot
                if (pred(ks_.decodeAt(id))) continue;   // dead — skip it
                if (write != read) {                // slide the survivor down
                    ks_.moveKeyTo(write, id);       // POD: value copy; byte-key:
                    ValueStore::moveValue(write + 1, id);   // location copy
                }
                ++write;
            }
            const int32_t removed = n - write;
            if (removed != 0) {
                ks_.truncate(write);
                ValueStore::truncateValues(write);
                rebuildIndex();
            }
            return removed;
        }

        // ---- Single-value MAP surface (SingleValueStore) -------------------
        // Each instantiates ONLY when called, so a set / multimap never
        // references the single-value primitives (on-demand member instantiation).

        /// @brief Insert a NEW key with its value (set-once).
        ///
        /// @details
        /// Mints the key (which must be new — re-inserting an existing key is a
        /// contract violation, asserted, never a silent overwrite) and appends
        /// the value at the matching id. Single-threaded write side only (I-83).
        /// A member template, so it instantiates only when called (a set never
        /// names a value).
        ///
        /// @tparam VV The value type (deduced; convertible to the value store's
        ///            value type).
        /// @param k The key; must not already be present.
        /// @param v The value to store.
        /// @return The key's id; >= 1.
        template <typename VV>
        int32_t insert(const KeyView& k, const VV& v) {
            const int32_t before = ks_.count();
            const int32_t id = mint(k);
            assert(id == before + 1
                && "HashMap::insert on an existing key (set-once — an in-place "
                   "value replace needs a reviewed PagedVector change)");
            ValueStore::appendValue(v);
            return id;
        }

        /// @brief Look up a key's value.
        ///
        /// @details
        /// A hit returns a pointer to the cold value (stable while resident); a
        /// miss returns `nullptr` — a defined query result (the bivalent twin of
        /// `lookup == 0`), not a failure fallback. Burst-safe (non-minting). The
        /// defaulted `VS` parameter defers instantiation to the call site, so a
        /// set never names a value type.
        ///
        /// @tparam VS The value store (defaulted; do not pass explicitly).
        /// @param k The key to look up.
        /// @return Pointer to the value, or `nullptr` when `k` is absent.
        template <typename VS = ValueStore>
        const typename VS::ValueType* find(const KeyView& k) const {
            const int32_t id = lookup(k);
            if (id == 0) return nullptr;
            return ValueStore::valuePtr(id);
        }

        /// @brief Value at a known id (single-value map).
        ///
        /// @tparam VS The value store (defaulted; do not pass explicitly).
        /// @param id A minted id; `1 <= id <= count()`.
        /// @return Const reference to the value.
        template <typename VS = ValueStore>
        const typename VS::ValueType& valueAt(int32_t id) const {
            assert(id >= 1 && id <= ks_.count()
                && "HashMap::valueAt on an unset id");
            return ValueStore::valueAt(id);
        }

        /// @brief Overwrite key `id`'s value in place (single-value map).
        ///
        /// @details
        /// The in-place update door the set-once `insert` forbids — a separate,
        /// reviewed entry point (Rule 8/19); `upsertStatementKey`'s flag-bit OR is
        /// its first consumer. Routes through `SingleValueStore::setValueAt` →
        /// `PagedVector::setAt`, which marks the aggregate `Restructured`. A member
        /// template via the defaulted `VS`, so a set never names it.
        /// Single-threaded write side only (I-83).
        ///
        /// @tparam VS The value store (defaulted; do not pass explicitly).
        /// @param id A minted id; `1 <= id <= count()`.
        /// @param v  The new value.
        template <typename VS = ValueStore>
        void setValueAt(int32_t id, const typename VS::ValueType& v) {
            assert(id >= 1 && id <= ks_.count()
                && "HashMap::setValueAt on an unset id");
            ValueStore::setValueAt(id, v);
        }

        /// @brief Overwrite key `id`'s value in place WITHOUT escalating the
        ///        dirty state — the parallel-safe twin of `setValueAt`
        ///        (single-value map).
        ///
        /// @details
        /// Routes through `SingleValueStore::setValueAtRelaxed` →
        /// `PagedVector::setAtRelaxed`: writes only the value slot, leaving the
        /// shared dirty flag untouched, so concurrent calls to DISJOINT ids are
        /// race-free. Sanctioned ONLY for a map on a NEVER-DELOADED pool whose
        /// dirty state is meaningless (it produces no deload image) — the
        /// pull-model mail cursor, advanced disjointly in the parallel phase-1
        /// pull. On a deloadable map it would silently skip the rewrite — a
        /// Rule-8/19 violation. A member template via the defaulted `VS`, so a
        /// set never names it.
        ///
        /// @tparam VS The value store (defaulted; do not pass explicitly).
        /// @param id A minted id; `1 <= id <= count()`.
        /// @param v  The new value.
        template <typename VS = ValueStore>
        void setValueAtRelaxed(int32_t id, const typename VS::ValueType& v) {
            assert(id >= 1 && id <= ks_.count()
                && "HashMap::setValueAtRelaxed on an unset id");
            ValueStore::setValueAtRelaxed(id, v);
        }

        // ---- Multi-value MAP surface (CsrValueStore) -----------------------

        /// @brief Append `v` to `k`'s run — append-to-tail only.
        ///
        /// @details
        /// `k` must be the current LAST key (extending its run) or a brand-new
        /// key (opening a fresh run at the current value end); an interior key
        /// asserts (a pure append, never an O(N) column shift). Single-threaded
        /// write side only (I-83). A member template, instantiated only when
        /// called.
        ///
        /// @tparam VV The value type (deduced).
        /// @param k The key; the last key or new.
        /// @param v The value to append.
        /// @return The key's id; >= 1.
        template <typename VV>
        int32_t appendToTail(const KeyView& k, const VV& v) {
            const int32_t before = ks_.count();
            const int32_t id = mint(k);
            if (id == before + 1) {
                ValueStore::openRun();      // brand-new key: open its run
            } else {
                assert(id == ks_.count()
                    && "HashMap::appendToTail on an interior key "
                       "(append-to-tail only — interior insert is deferred)");
            }
            ValueStore::appendValue(v);
            return id;
        }

        /// @brief Total number of values across every run (multi-value map).
        ///
        /// @return Value count.
        int32_t valueCount() const { return ValueStore::valueCount(); }

        /// @brief Number of values in key `id`'s run — DERIVED from the CSR
        ///        offsets (next start, or the total for the last key).
        ///
        /// @param id A minted id; `1 <= id <= count()`.
        /// @return The run length.
        int32_t runLen(int32_t id) const {
            assert(id >= 1 && id <= ks_.count()
                && "HashMap::runLen on an unset id");
            return ValueStore::runLen(id, ks_.count());
        }

        /// @brief Value `j` of key `id`'s run (multi-value map).
        ///
        /// @tparam VS The value store (defaulted; do not pass explicitly).
        /// @param id A minted id; `1 <= id <= count()`.
        /// @param j  Position in `[0, runLen(id))`.
        /// @return Const reference to the value.
        template <typename VS = ValueStore>
        const typename VS::ValueType& valueAt(int32_t id, int32_t j) const {
            assert(j >= 0 && j < runLen(id)
                && "HashMap::valueAt out of run range");
            return ValueStore::valueAt(id, j);
        }

        // ---- SET-MAP surface (SetValueStore) -------------------------------
        // Each references SetValueStore-only splice primitives (runStartRaw /
        // valueRaw / insertValueAt / ...), so it instantiates ONLY for a
        // `ColdSetMap` (a `ColdMultiMap`'s `CsrValueStore` lacks them — naming
        // these on a multimap is a compile error, the intended "use insertSorted
        // only on a set map"). The run stays SORTED + DUPLICATE-FREE under `Cmp`.

        /// @brief Insert `v` into key `k`'s sorted-unique run — the set-map
        ///        insert (interior growth `appendToTail` cannot do).
        ///
        /// @details
        /// Find-or-mints `k`. A brand-new key opens a fresh run holding `v`
        /// alone (O(1) amortized). An existing key binary-searches its run with
        /// `cmp`, returns unchanged on a `cmp`-equal hit (the set dedup), else
        /// splices `v` at its sorted position (`insertValueAt`, an
        /// O(value-tail) column shift) and bumps every later key's run start by
        /// one (O(key-tail)). The predicate is passed per call, never stored —
        /// the caller holds it (the `ValueInterner`-bound decoded-id comparator
        /// for `orBookkeeping`; the default `std::less<V>` natural order for the
        /// packed-int level maps). Single-threaded write side only (I-83).
        ///
        /// @tparam VS  The value store (defaulted; do not pass explicitly — it
        ///             keeps `ValueType` dependent so a set never names it).
        /// @tparam Cmp A strict-weak-ordering callable `bool(const V&, const V&)`.
        /// @param k   The key.
        /// @param v   The value to insert.
        /// @param cmp The run's ordering predicate (defaulted to `std::less<V>`).
        /// @return The key's id; >= 1.
        template <typename VS = ValueStore,
                  typename Cmp = std::less<typename VS::ValueType>>
        int32_t insertSorted(const KeyView& k,
                             const typename VS::ValueType& v,
                             Cmp cmp = Cmp{}) {
            const int32_t before = ks_.count();
            const int32_t id = mint(k);
            if (id == before + 1) {          // brand-new key: open run + 1 value
                ValueStore::openRun();
                ValueStore::appendValue(v);
                return id;
            }
            const int32_t rs = ValueStore::runStartRaw(id - 1);
            const int32_t rl = ValueStore::runLen(id, ks_.count());
            int32_t lo = 0, hi = rl;         // lower_bound over the sorted run
            while (lo < hi) {
                const int32_t mid = lo + ((hi - lo) >> 1);
                if (cmp(ValueStore::valueRaw(rs + mid), v)) lo = mid + 1;
                else hi = mid;
            }
            if (lo < rl && !cmp(v, ValueStore::valueRaw(rs + lo)))
                return id;                   // cmp-equal -> already present
            ValueStore::insertValueAt(rs + lo, v);
            const int32_t n = ks_.count();
            for (int32_t idx = id; idx < n; ++idx)   // shift later run starts +1
                ValueStore::setRunStartRaw(idx, ValueStore::runStartRaw(idx) + 1);
            return id;
        }

        /// @brief Replace key `k`'s run with the `m` already-sorted-unique
        ///        values `vals` — the whole-set replace (`intStatementLevelsMap`'s
        ///        write-once-or-overwrite door).
        ///
        /// @details
        /// Find-or-mints `k`. A brand-new key opens a run and appends the values
        /// verbatim (the caller supplies them in the run's order — a
        /// `std::set` iteration is ascending, matching the default order). An
        /// existing key overwrites the common prefix in place (no shift when the
        /// size is unchanged — the dominant case) then grows (`insertValueAt`)
        /// or shrinks (`eraseValueAt`) the tail and adjusts every later key's
        /// run start by the size delta. Single-threaded write side only (I-83).
        ///
        /// @tparam VS  The value store (defaulted; do not pass explicitly — it
        ///             keeps `ValueType` dependent so a set never names it).
        /// @param k    The key.
        /// @param vals The new run values, pre-sorted + unique; read only when
        ///             `m > 0`.
        /// @param m    Value count; >= 0.
        /// @return The key's id; >= 1.
        template <typename VS = ValueStore>
        int32_t assignSet(const KeyView& k,
                          const typename VS::ValueType* vals,
                          int32_t m) {
            assert(m >= 0 && (m == 0 || vals != nullptr));
#ifndef NDEBUG
            for (int32_t j = 1; j < m; ++j)
                assert(vals[j - 1] < vals[j]
                    && "assignSet values must be sorted + unique (a set run)");
#endif
            const int32_t before = ks_.count();
            const int32_t id = mint(k);
            if (id == before + 1) {          // brand-new key
                ValueStore::openRun();
                for (int32_t j = 0; j < m; ++j) ValueStore::appendValue(vals[j]);
                return id;
            }
            const int32_t rs = ValueStore::runStartRaw(id - 1);
            const int32_t rl = ValueStore::runLen(id, ks_.count());
            const int32_t common = (rl < m) ? rl : m;
            for (int32_t j = 0; j < common; ++j)
                ValueStore::setValueRaw(rs + j, vals[j]);
            const int32_t n = ks_.count();
            if (m > rl) {
                for (int32_t j = rl; j < m; ++j)
                    ValueStore::insertValueAt(rs + j, vals[j]);
                const int32_t delta = m - rl;
                for (int32_t idx = id; idx < n; ++idx)
                    ValueStore::setRunStartRaw(
                        idx, ValueStore::runStartRaw(idx) + delta);
            } else if (m < rl) {
                for (int32_t j = m; j < rl; ++j)
                    ValueStore::eraseValueAt(rs + m);   // remove at rs+m each time
                const int32_t delta = rl - m;
                for (int32_t idx = id; idx < n; ++idx)
                    ValueStore::setRunStartRaw(
                        idx, ValueStore::runStartRaw(idx) - delta);
            }
            return id;
        }

        /// @brief Replace key `k`'s run with the sorted-unique values in
        ///        `[first, last)` — the `std::set` / range convenience over
        ///        `assignSet` (the level maps' `[pk] = levelSet` write door).
        ///
        /// @details
        /// Copies the range into a transient heap buffer (like the reshuffle
        /// scratch), sorts + dedups it into the canonical set order, then
        /// forwards to `assignSet`. A `std::set` source is already sorted-unique
        /// (the sort/unique are no-ops); a `std::vector` source is normalized,
        /// so the run is always a valid set regardless of the caller's order.
        ///
        /// @tparam VS  The value store (defaulted; do not pass explicitly).
        /// @tparam It  A forward iterator over the value type.
        /// @param k     The key.
        /// @param first Range begin.
        /// @param last  Range end.
        /// @return The key's id; >= 1.
        template <typename VS = ValueStore, typename It>
        int32_t assignSetRange(const KeyView& k, It first, It last) {
            std::vector<typename VS::ValueType> tmp(first, last);
            std::sort(tmp.begin(), tmp.end());
            tmp.erase(std::unique(tmp.begin(), tmp.end()), tmp.end());
            return assignSet(k, tmp.data(), static_cast<int32_t>(tmp.size()));
        }

        /// @brief Whether key `k`'s run contains `v` — the set-membership probe.
        ///
        /// @details
        /// A non-minting lookup then a linear scan of the run (runs are small —
        /// level / disjunct / auxie sets). Burst-safe (read-only). `V` is
        /// compared with `operator==` (the packed ints / ids these maps hold).
        ///
        /// @tparam VS The value store (defaulted; do not pass explicitly).
        /// @param k The key.
        /// @param v The value to test.
        /// @return `true` when `k` is present and its run contains `v`.
        template <typename VS = ValueStore>
        bool setContains(const KeyView& k,
                         const typename VS::ValueType& v) const {
            const int32_t id = lookup(k);
            if (id == 0) return false;
            const int32_t rs = ValueStore::runStartRaw(id - 1);
            const int32_t rl = ValueStore::runLen(id, ks_.count());
            for (int32_t j = 0; j < rl; ++j)
                if (ValueStore::valueRaw(rs + j) == v) return true;
            return false;
        }

        /// @brief Remove key `k` and its whole run — the set-map per-key erase.
        ///
        /// @param k The key to remove.
        /// @return `true` when a key was removed, `false` on a miss.
        bool eraseSet(const KeyView& k) {
            return eraseSetIf([&k](const auto& kd) { return kd == k; }) != 0;
        }

        /// @brief Remove every key the predicate accepts — the run-aware
        ///        compacting scope-wipe (the set-map `wipeSubtree` driver).
        ///
        /// @details
        /// A single forward COMPACTION pass: survivors slide their key, their
        /// run start, and their whole value run to the front in lockstep
        /// (`setKeyAt` / `setRunStartRaw` / `setValueRaw`), the dead tails are
        /// `truncate`d, and the throw-away index is rebuilt once. O(count() +
        /// valueCount()) with one rebuild — survivor order preserved, so the
        /// post-wipe ids and deload bytes are a deterministic function of the
        /// surviving content (history invisible). Every write is guarded against
        /// a no-op, so a predicate that matches nothing leaves the container —
        /// and its dirty state — untouched. POD-key only (`setKeyAt`).
        /// Single-threaded write side only (I-83).
        ///
        /// @tparam Pred A callable `bool(KeyDecode)` — erase the key when true.
        /// @param pred The erase predicate, tested against each decoded key.
        /// @return Number of keys removed.
        template <typename Pred>
        int32_t eraseSetIf(Pred pred) {
            assert(arena_->resident()
                && "eraseSetIf on a deloaded LB (I-111)");
            const int32_t n = ks_.count();
            int32_t writeKey = 0;   // 0-based survivor key cursor
            int32_t writeVal = 0;   // 0-based survivor value-column cursor
            for (int32_t read = 1; read <= n; ++read) {
                const int32_t rs = ValueStore::runStartRaw(read - 1);
                const int32_t rl = ValueStore::runLen(read, n);
                if (pred(ks_.decodeAt(read))) continue;   // drop key + its run
                if (writeKey != read - 1)
                    ks_.setKeyAt(writeKey, ks_.decodeAt(read));
                if (ValueStore::runStartRaw(writeKey) != writeVal)
                    ValueStore::setRunStartRaw(writeKey, writeVal);
                for (int32_t j = 0; j < rl; ++j) {
                    if (writeVal != rs + j)
                        ValueStore::setValueRaw(writeVal,
                                                ValueStore::valueRaw(rs + j));
                    ++writeVal;
                }
                ++writeKey;
            }
            const int32_t removed = n - writeKey;
            if (removed != 0) {
                ks_.truncate(writeKey);
                ValueStore::truncateRunStarts(writeKey);
                ValueStore::truncateValues(writeVal);
                rebuildIndex();
            }
            return removed;
        }

        // ---- BLOB-MAP surface (BlobCsrValueStore) --------------------------
        // Each references BlobCsrValueStore-only primitives (appendBlob /
        // replaceRun / runByteStart / moveBytes / ...), so it instantiates ONLY
        // for a `ColdBlobMap`. The value is opaque bytes — one record's canonical
        // serialization produced by a per-record codec at the call site; the
        // store never interprets it.

        /// @brief Total blobs across every run (blob map).
        ///
        /// @return Blob count — the blob-start tag's element count.
        int32_t blobCount() const { return ValueStore::blobCount(); }

        /// @brief Total blob-pool bytes (blob map) — the pool tag's element count.
        ///
        /// @return Pool byte count.
        int32_t poolByteCount() const { return ValueStore::poolByteCount(); }

        /// @brief Replace key `k`'s run with `M` blobs — the whole-run replace
        ///        (the `equivalenceClassesMap` bucket-rebuild write door).
        ///
        /// @details
        /// Find-or-mints `k`. A brand-new key opens a run and appends the `M`
        /// blobs verbatim (the caller supplies them in the run's intended order —
        /// the equivalence-class list order, NOT sorted). An existing key splices
        /// its run in place (`BlobCsrValueStore::replaceRun`, one byte-tail + one
        /// blob-tail pass) then this rebases every later key's run start by the
        /// blob-count delta (O(key-tail)). The blobs' bytes are concatenated in
        /// `bytes` with per-blob lengths in `lens`. Single-threaded write side
        /// only (I-83). A member template, instantiated only when called.
        ///
        /// @param k     The key.
        /// @param bytes The `M` blobs' bytes concatenated; read only when the
        ///              total is positive.
        /// @param lens  The `M` blobs' per-blob byte lengths.
        /// @param M     Blob count; >= 0.
        /// @return The key's id; >= 1.
        template <typename VS = ValueStore>
        int32_t assignRun(const KeyView& k, const char* bytes,
                          const int32_t* lens, int32_t M) {
            assert(M >= 0 && (M == 0 || lens != nullptr));
            const int32_t before = ks_.count();
            int32_t id = 0;
            {
                RT_SCOPE_HERE("CSR_MINT_KEY");
                id = mint(k);
            }
            if (id == before + 1) {              // brand-new key
                RT_SCOPE_HERE("CSR_APPEND_RUN");
                ValueStore::openRun();
                int32_t off = 0;
                for (int32_t j = 0; j < M; ++j) {
                    ValueStore::appendBlob(bytes + off, lens[j]);
                    off += lens[j];
                }
                return id;
            }
            return assignRunAtId(id, bytes, lens, M);
        }

        /// @brief Replace key @p k's blob run from a replayable segmented record
        ///        emitter, with one byte-tail move and one blob-tail move.
        ///
        /// @details
        /// The non-contiguous twin of @ref assignRun. A new key opens at the pool
        /// tail and appends emitted records. An existing key delegates to
        /// `BlobCsrValueStore::replaceRunGenerated`, then rebases later key-run
        /// starts once by the blob-count delta. The resulting dense CSR bytes are
        /// identical to concatenating the same records and calling @ref assignRun,
        /// but the source may span arbitrarily many arena blocks.
        ///
        /// @tparam Emit Replayable callable accepting a record sink with signature
        ///              `void(const char* bytes, int32_t len)`.
        /// @tparam VS Value-store selector; instantiated only for
        ///            `BlobCsrValueStore`.
        /// @param k        Key view to find or mint.
        /// @param M        Number of emitted records; >= 0.
        /// @param newBytes Sum of emitted record lengths; >= 0.
        /// @param emit     Producer of the final run in canonical order.
        /// @return The key id; >= 1.
        /// @invariant Key ids and final run bytes match @ref assignRun for the
        ///            same ordered record sequence.
        /// @see BlobCsrValueStore::replaceRunGenerated.
        template <typename Emit, typename VS = ValueStore>
        int32_t assignRunGenerated(const KeyView& k, int32_t M,
                                   int32_t newBytes, Emit emit) {
            assert(M >= 0 && newBytes >= 0);
            const int32_t before = ks_.count();
            int32_t id = 0;
            {
                RT_SCOPE_HERE("CSR_MINT_KEY");
                id = mint(k);
            }
            if (id == before + 1) {
                RT_SCOPE_HERE("CSR_APPEND_RUN");
                ValueStore::openRun();
                int32_t emitted = 0, bytesEmitted = 0;
                emit([&](const char* bytes, int32_t len) {
                    assert(len >= 0 && (len == 0 || bytes != nullptr));
                    ValueStore::appendBlob(bytes, len);
                    ++emitted;
                    bytesEmitted += len;
                });
                assert(emitted == M && bytesEmitted == newBytes
                    && "HashMap::assignRunGenerated new-run emitter shape "
                       "mismatch");
                return id;
            }
            return assignRunGeneratedAtId(id, M, newBytes, emit);
        }

        /// @brief Replace an EXISTING key id's blob run without probing its key
        ///        again.
        ///
        /// @details
        /// The known-hit twin of @ref assignRun. A caller that has just obtained
        /// @p id from `lookup(k)` already proved key membership; routing the
        /// subsequent whole-run write through `mint(k)` would hash and probe the
        /// same key a second time. This door starts at the stored run descriptor,
        /// performs the identical dense CSR replacement, and rebases later key
        /// starts once. It cannot mint and therefore cannot change key ids.
        ///
        /// @tparam VS Value-store selector; instantiated only for
        ///            `BlobCsrValueStore`.
        /// @param id    Existing key id; `1 <= id <= count()`.
        /// @param bytes Concatenated replacement blob bytes.
        /// @param lens  Replacement blob lengths.
        /// @param M     Replacement blob count; >= 0.
        /// @return The unchanged @p id.
        /// @invariant Final value columns are byte-identical to
        ///            `assignRun(decode(id), bytes, lens, M)`.
        /// @see assignRun, assignRunGeneratedAtId.
        template <typename VS = ValueStore>
        int32_t assignRunAtId(int32_t id, const char* bytes,
                              const int32_t* lens, int32_t M) {
            const int32_t keyCount = ks_.count();
            assert(id >= 1 && id <= keyCount
                && "HashMap::assignRunAtId on an unstored id");
            assert(M >= 0 && (M == 0 || lens != nullptr));
            const int32_t b0 = ValueStore::runStartRaw(id - 1);
            const int32_t oldCount = ValueStore::runLen(id, keyCount);
            {
                RT_SCOPE_HERE("CSR_REPLACE_RUN");
                ValueStore::replaceRun(b0, oldCount, bytes, lens, M);
            }
            const int32_t countDelta = M - oldCount;
            if (countDelta != 0) {
                RT_SCOPE_HERE("CSR_RUNSTARTS_SHIFT");
                RT_NOTE_ITERATIONS_HERE(keyCount - id);   // later keys renumbered
                ValueStore::addToRunStartsSuffix(id, countDelta);
            }
            return id;
        }

        /// @brief Replace an EXISTING key id's blob run from a replayable emitter
        ///        without probing its key again.
        ///
        /// @details
        /// The generated-source twin of @ref assignRunAtId and known-hit twin of
        /// @ref assignRunGenerated. It preserves the one-tail-move segmented CSR
        /// write while eliminating the redundant `mint`/`lookup` performed after
        /// the caller has already resolved the key id.
        ///
        /// @tparam Emit Replayable callable accepting a record sink with signature
        ///              `void(const char* bytes, int32_t len)`.
        /// @tparam VS Value-store selector; instantiated only for
        ///            `BlobCsrValueStore`.
        /// @param id       Existing key id; `1 <= id <= count()`.
        /// @param M        Replacement blob count; >= 0.
        /// @param newBytes Sum of emitted record lengths; >= 0.
        /// @param emit     Producer of the final run in canonical order.
        /// @return The unchanged @p id.
        /// @invariant Final value columns are byte-identical to
        ///            `assignRunGenerated(decode(id), M, newBytes, emit)`.
        /// @see assignRunGenerated, assignRunAtId.
        template <typename Emit, typename VS = ValueStore>
        int32_t assignRunGeneratedAtId(int32_t id, int32_t M,
                                       int32_t newBytes, Emit emit) {
            const int32_t keyCount = ks_.count();
            assert(id >= 1 && id <= keyCount
                && "HashMap::assignRunGeneratedAtId on an unstored id");
            assert(M >= 0 && newBytes >= 0);
            const int32_t b0 = ValueStore::runStartRaw(id - 1);
            const int32_t oldCount = ValueStore::runLen(id, keyCount);
            {
                RT_SCOPE_HERE("CSR_REPLACE_RUN");
                ValueStore::replaceRunGenerated(b0, oldCount, M, newBytes, emit);
            }
            const int32_t countDelta = M - oldCount;
            if (countDelta != 0) {
                RT_SCOPE_HERE("CSR_RUNSTARTS_SHIFT");
                RT_NOTE_ITERATIONS_HERE(keyCount - id);   // later keys renumbered
                ValueStore::addToRunStartsSuffix(id, countDelta);
            }
            return id;
        }

        /// @brief Append ONE blob to an existing key `id`'s run-end WITHOUT
        ///        touching the existing blobs — the O(blob-tail) RMW append
        ///        fast path.
        ///
        /// @details
        /// The hot insert path (`addToHashMemory`'s encodedMap RMW) used to grow
        /// a key's run by reading the whole run (`recordsAt`), pushing one
        /// record, and re-`assignRun`ing it — O(run) decode + O(run) re-encode
        /// per insert, ~O(run^2) over a key's lifetime. This instead splices the
        /// single new blob at the run-end (`replaceRun` with `oldCount == 0`, an
        /// insert) and rebases every later key's run start by one; the existing
        /// blobs are never decoded or re-encoded, so the per-insert cost drops to
        /// O(blob-tail) (O(1) amortized when the key owns the pool tail). The
        /// resulting run content is BYTE-IDENTICAL to the former
        /// read-modify-`assignRun` — the new blob lands at the same run-end
        /// position. The key MUST already exist (the caller checks via `lookup`;
        /// a brand-new key still goes through `assignRun`). Single-threaded write
        /// side only (I-83). A member template, instantiated only when called.
        ///
        /// @param id   An existing key's id; `1 <= id <= count()`.
        /// @param blob The one new blob's bytes; read only when `len > 0`.
        /// @param len  The blob's byte length; >= 0.
        template <typename VS = ValueStore>
        void appendBlobToRun(int32_t id, const char* blob, int32_t len) {
            const int32_t kc = ks_.count();
            assert(id >= 1 && id <= kc
                && "HashMap::appendBlobToRun on a non-existent key");
            const int32_t b0 = ValueStore::runStartRaw(id - 1);
            const int32_t oldCount = ValueStore::runLen(id, kc);
            // Insert the one new blob at the run-end (oldCount == 0 -> a pure
            // insert; the existing blobs are not read).
            {
                RT_SCOPE_HERE("CSR_REPLACE_RUN");
                ValueStore::replaceRun(b0 + oldCount, 0, blob, &len, 1);
            }
            // One blob was inserted before every later key's run -> +1 each.
            {
                RT_SCOPE_HERE("CSR_RUNSTARTS_SHIFT");
                RT_NOTE_ITERATIONS_HERE(kc - id);   // later keys renumbered
                ValueStore::addToRunStartsSuffix(id, 1);
            }
        }

        /// @brief Replace EVERY key's run at once from a per-key record emitter
        ///        — the batch twin of a sequence of `assignRun` /
        ///        `appendBlobToRun` splices (blob map).
        ///
        /// @details
        /// The emitter is called for every id in `1..keyCount` and may read the
        /// current runs of the stored keys (`forEachBlobContiguous`,
        /// `peekBlobContiguous`, ids `<= count()`) while producing — the
        /// columns are rebuilt only after the last key
        /// (`BlobCsrValueStore::rebuildRunsGenerated`). Ids beyond `count()`
        /// are the keys the caller mints RIGHT AFTER this call, in id order
        /// (the first mint receives `count() + 1`, and so on); minting them
        /// before would put the run-start column and the key store out of
        /// step for the emitter's reads. The key store and the index are
        /// untouched here. Single-threaded write side only (I-83). A member
        /// template, instantiated only when called.
        ///
        /// @tparam EmitAll Callable `void(int32_t id, RebuildSink& sink)` —
        ///                 `sink.blob(bytes, len)` per record in final run
        ///                 order, or `sink.unchanged(id)` for a stored key
        ///                 whose run does not change.
        /// @param keyCount The key count after the caller's pending mints;
        ///                 `>= count()`.
        /// @param emitAll  The per-key record producer.
        /// @param scratch  Per-slot scratch arena for the rebuild's collection
        ///                 columns.
        /// @return Nothing.
        /// @invariant The value columns equal those of a fresh map given the
        ///            same keys and the same final runs.
        /// @see BlobCsrValueStore::rebuildRunsGenerated, assignRun, appendBlobToRun.
        template <typename EmitAll, typename VS = ValueStore>
        void assignAllRunsGenerated(int32_t keyCount, EmitAll emitAll,
                                    LbArena& scratch) {
            assert(arena_->resident()
                && "assignAllRunsGenerated on a deloaded LB (I-111)");
            assert(keyCount >= ks_.count()
                && "assignAllRunsGenerated: fewer runs than stored keys");
            ValueStore::rebuildRunsInPlace(keyCount, emitAll, scratch);
        }

        /// @brief The scratch-rebuild twin of @ref assignAllRunsGenerated —
        ///        the test oracle (`BlobCsrValueStore::rebuildRunsGenerated`,
        ///        one write pass into scratch columns, then a refill).
        ///
        /// @tparam EmitAll As for @ref assignAllRunsGenerated (one pass).
        /// @param keyCount As for @ref assignAllRunsGenerated.
        /// @param emitAll  The per-key record producer.
        /// @param scratch  Scratch arena for the three collection columns.
        /// @return Nothing.
        template <typename EmitAll, typename VS = ValueStore>
        void assignAllRunsViaScratch(int32_t keyCount, EmitAll emitAll,
                                     LbArena& scratch) {
            assert(arena_->resident()
                && "assignAllRunsViaScratch on a deloaded LB (I-111)");
            assert(keyCount >= ks_.count()
                && "assignAllRunsViaScratch: fewer runs than stored keys");
            ValueStore::rebuildRunsGenerated(keyCount, emitAll, scratch);
        }

        /// @brief Copy blob `j` of key `id`'s run into `out` (blob map).
        ///
        /// @details
        /// The read boundary the codec deserializes from — a heap copy because a
        /// blob may straddle pages. Burst-safe (read-only). A member template,
        /// instantiated only when called.
        ///
        /// @param id  A minted id; `1 <= id <= count()`.
        /// @param j   Position in `[0, runLen(id))`.
        /// @param out Destination buffer (resized to the blob length).
        template <typename VS = ValueStore>
        void blobAt(int32_t id, int32_t j, std::vector<char>& out) const {
            assert(j >= 0 && j < runLen(id)
                && "HashMap::blobAt out of run range");
            ValueStore::readBlob(ValueStore::runStartRaw(id - 1) + j, out);
        }

        /// @brief Copy blob `j` of key `id`'s run into the caller-provided
        ///        buffer `out` (blob map) — the no-heap twin of the
        ///        `std::vector<char>` `blobAt`.
        ///
        /// @details
        /// Forwards to the value store's raw-pointer `readBlob`. The caller must
        /// have reserved at least the blob's byte length at `out`; the
        /// arena-backed `peekBlobContiguous` reserves exactly the length
        /// `peekBlobAt` returns. A member template, instantiated only when
        /// called.
        ///
        /// @param id  A minted id; `1 <= id <= count()`.
        /// @param j   Position in `[0, runLen(id))`.
        /// @param out Destination buffer, at least the blob's byte length.
        template <typename VS = ValueStore>
        void blobAt(int32_t id, int32_t j, char* out) const {
            assert(j >= 0 && j < runLen(id)
                && "HashMap::blobAt out of run range");
            ValueStore::readBlob(ValueStore::runStartRaw(id - 1) + j, out);
        }

        /// @brief Zero-copy peek of blob `j` of key `id`'s run — the
        ///        no-allocation twin of `blobAt` (blob map).
        ///
        /// @details
        /// Resolves the run's blob index and forwards to the value store's
        /// contiguous peek (`BlobCsrValueStore::peekBlob`). Read-only; burst-safe.
        /// A member template, instantiated only when called.
        ///
        /// @param id  A minted id; `1 <= id <= count()`.
        /// @param j   Position in `[0, runLen(id))`.
        /// @param p   [out] Pointer to the blob's bytes (valid for `len` bytes)
        ///            when the result is `true`.
        /// @param len [out] The blob's byte length (always set).
        /// @return `true` when `p` spans the whole blob contiguously; `false` on a
        ///         page straddle (the caller must `blobAt` instead).
        template <typename VS = ValueStore>
        bool peekBlobAt(int32_t id, int32_t j, const char*& p,
                        int32_t& len) const {
            assert(j >= 0 && j < runLen(id)
                && "HashMap::peekBlobAt out of run range");
            return ValueStore::peekBlob(ValueStore::runStartRaw(id - 1) + j,
                                        p, len);
        }

        /// @brief Contiguous bytes of blob `j` of key `id`'s run — zero-copy when
        ///        the blob is single-page, else copied into `scratch` once (blob
        ///        map).
        ///
        /// @details
        /// The owner-set prune's read door: returns a pointer to `len` contiguous
        /// blob bytes the caller parses field-by-field. The common case (a small
        /// record on one page) returns a pointer straight into the arena pool and
        /// never touches `scratch`; only a page-straddling blob falls back to the
        /// copying `blobAt(id, j, scratch)`. So the hot path pays no per-probe heap
        /// allocation on the common case — the whole point of the peek. Read-only;
        /// burst-safe. A member template, instantiated only when called.
        ///
        /// @param id      A minted id; `1 <= id <= count()`.
        /// @param j       Position in `[0, runLen(id))`.
        /// @param len     [out] The blob's byte length.
        /// @param scratch Caller-owned reuse buffer (e.g. thread-local) used only
        ///                on a straddle; left untouched on the contiguous case.
        /// @return Pointer to `len` contiguous blob bytes (into the pool, or into
        ///         `scratch` on a straddle).
        template <typename VS = ValueStore>
        const char* peekBlobContiguous(int32_t id, int32_t j, int32_t& len,
                                       std::vector<char>& scratch) const {
            const char* p = nullptr;
            if (peekBlobAt(id, j, p, len)) return p;
            blobAt(id, j, scratch);
            len = static_cast<int32_t>(scratch.size());
            return scratch.data();
        }

        /// @brief Contiguous bytes of blob `j` of key `id`'s run — zero-copy
        ///        when single-page, else assembled once onto the arena's
        ///        byte-bump tier (the arena-backed twin of the
        ///        `std::vector<char>` `peekBlobContiguous`, blob map).
        ///
        /// @details
        /// The recursion-safe read door for the equivalence-class processing
        /// path. The common case (a class record on one page) returns a pointer
        /// straight into the pool and never touches `scratch`; only a
        /// page-straddling record is copied — once — into a FRESH
        /// `scratch.alloc(len, 1)` byte-bump run. Because each straddle takes a
        /// fresh allocation with no rewind, an outer peek's bytes survive a
        /// nested call that itself peeks (the
        /// `applyEquivalenceClassToNegatedEquality` -> `addStatement` -> self
        /// recursion): straddle buffers only accumulate, never overwrite one
        /// another, until the per-task `releaseAll` reclaims the arena. `len` is
        /// taken from `peekBlobAt` (which sets it, always, before its contiguity
        /// test), so the reservation is exactly the blob's byte length and the
        /// returned bytes are byte-identical to the `std::vector<char>` overload.
        /// Read-only; burst-safe. A member template, instantiated only when
        /// called.
        ///
        /// @param id      A minted id; `1 <= id <= count()`.
        /// @param j       Position in `[0, runLen(id))`.
        /// @param len     [out] The blob's byte length.
        /// @param scratch Byte-bump arena the straddle copy is assembled onto;
        ///                left untouched on the contiguous case.
        /// @return Pointer to `len` contiguous blob bytes (into the pool, or
        ///         into `scratch` on a straddle).
        template <typename VS = ValueStore>
        const char* peekBlobContiguous(int32_t id, int32_t j, int32_t& len,
                                       ScratchArena& scratch) const {
            const char* p = nullptr;
            if (peekBlobAt(id, j, p, len)) return p;
            char* buf = scratch.resolve(scratch.alloc(len, 1));
            blobAt(id, j, buf);
            return buf;
        }

        /// @brief Visit every blob in key @p id's run through the value store's
        ///        page-cached sequential scanner.
        ///
        /// @details
        /// Resolves the run start and length once, then delegates to
        /// `BlobCsrValueStore::forEachBlobRange`. Compared with calling
        /// `peekBlobContiguous(id, j, ...)` in a loop, adjacent blob-start entries
        /// and byte spans reuse their current pages instead of re-entering both
        /// paged directories for every record. Read-only and burst-safe.
        ///
        /// @tparam Fn Callable `void(const char* bytes, int32_t len)`.
        /// @tparam VS Value-store selector; instantiated only for
        ///            `BlobCsrValueStore`.
        /// @param id      A minted id; `1 <= id <= count()`.
        /// @param scratch Arena used only when a blob straddles a pool page.
        /// @param fn      Consumer invoked once per record in run order.
        /// @return Nothing.
        /// @invariant The callback sequence is byte-identical to calling
        ///            `blobAt(id, j)` for every `j` in the run.
        /// @see BlobCsrValueStore::forEachBlobRange.
        template <typename Fn, typename VS = ValueStore>
        void forEachBlobContiguous(int32_t id, ScratchArena& scratch,
                                   Fn fn) const {
            const int32_t keyCount = ks_.count();
            assert(id >= 1 && id <= keyCount
                && "HashMap::forEachBlobContiguous on an unstored id");
            const int32_t b0 = ValueStore::runStartRaw(id - 1);
            const int32_t count = ValueStore::runLen(id, keyCount);
            ValueStore::forEachBlobRange(b0, count, scratch, fn);
        }

        /// @brief Contiguous bytes of blob `j` of key `id`'s run — zero-copy
        ///        when single-page, else assembled once into the CALLER's
        ///        fixed-capacity buffer (the stack-buffer twin of the
        ///        `std::vector<char>` / `ScratchArena&` `peekBlobContiguous`
        ///        overloads, blob map).
        ///
        /// @details
        /// The no-arena read door for RMWs whose blobs have a closed-form
        /// size ceiling (the origin history lines): the common case (a small
        /// record on one page) returns a pointer straight into the pool and
        /// never touches `buf`; only a page-straddling blob is copied — once
        /// — into `buf`. The capacity check is a hard `assert` naming the
        /// contract (Rule 19): a blob larger than the caller's ceiling is a
        /// broken size invariant at the write side, never something to clamp.
        /// `len` is taken from `peekBlobAt` (which sets it, always, before
        /// its contiguity test), so the returned bytes are byte-identical to
        /// the `std::vector<char>` overload's. Read-only; burst-safe. A
        /// member template, instantiated only when called.
        ///
        /// @param id  A minted id; `1 <= id <= count()`.
        /// @param j   Position in `[0, runLen(id))`.
        /// @param len [out] The blob's byte length.
        /// @param buf Caller-owned buffer of at least `cap` bytes; written
        ///            only on a straddle.
        /// @param cap The buffer's capacity; a straddling blob longer than
        ///            this asserts.
        /// @return Pointer to `len` contiguous blob bytes (into the pool, or
        ///         into `buf` on a straddle).
        /// @see peekBlobContiguous(int32_t, int32_t, int32_t&, std::vector<char>&)
        ///      — the heap oracle the twin test compares against.
        template <typename VS = ValueStore>
        const char* peekBlobContiguous(int32_t id, int32_t j, int32_t& len,
                                       char* buf, int32_t cap) const {
            const char* p = nullptr;
            if (peekBlobAt(id, j, p, len)) return p;
            assert(len <= cap
                && "HashMap::peekBlobContiguous: straddling blob exceeds the "
                   "caller buffer capacity — widen the caller's constant "
                   "(Rule-19 tripwire)");
            blobAt(id, j, buf);
            return buf;
        }

        /// @brief Remove every key the predicate accepts — the run-aware
        ///        compacting scope-wipe for the blob map (the `wipeSubtree`
        ///        driver).
        ///
        /// @details
        /// A single forward COMPACTION pass over THREE value columns: survivors
        /// slide their key, run start, the run's blob-start entries (byte offsets
        /// rebased to the survivor pool cursor), and the run's pool bytes to the
        /// front in lockstep; the dead tails are `truncate`d and the index rebuilt
        /// once. O(count() + blobCount() + poolBytes) with one rebuild — survivor
        /// order preserved, so the post-wipe ids and deload bytes are a
        /// deterministic function of the surviving content. Every write is guarded
        /// against a no-op, so a predicate matching nothing leaves the container —
        /// and its dirty state — untouched. POD and byte keys alike (`moveKeyTo`: a
        /// byte key slides only its location entry, its dead bytes stay as a hole
        /// the copying compaction reclaims — I-119).
        /// Single-threaded write side only (I-83).
        ///
        /// @tparam Pred A callable `bool(KeyDecode)` — erase the key when true.
        /// @param pred The erase predicate, tested against each decoded key.
        /// @return Number of keys removed.
        template <typename Pred>
        int32_t eraseBlobIf(Pred pred) {
            assert(arena_->resident()
                && "eraseBlobIf on a deloaded LB (I-111)");
            const int32_t n = ks_.count();
            int32_t writeKey = 0, writeBlob = 0, writeByte = 0;
            for (int32_t read = 1; read <= n; ++read) {
                const int32_t rs = ValueStore::runStartRaw(read - 1);
                const int32_t rl = ValueStore::runLen(read, n);
                const int32_t yStart = ValueStore::runByteStart(rs);
                const int32_t runBytes =
                    ValueStore::runByteStart(rs + rl) - yStart;
                if (pred(ks_.decodeAt(read))) continue;   // drop key + its run
                if (writeKey != read - 1)
                    ks_.moveKeyTo(writeKey, read);
                if (ValueStore::runStartRaw(writeKey) != writeBlob)
                    ValueStore::setRunStartRaw(writeKey, writeBlob);
                for (int32_t j = 0; j < rl; ++j) {
                    const int32_t oldB = rs + j;
                    const int32_t newStart =
                        writeByte + (ValueStore::runByteStart(oldB) - yStart);
                    if (writeBlob != oldB
                        || ValueStore::runByteStart(writeBlob) != newStart)
                        ValueStore::setBlobStartRaw(writeBlob, newStart);
                    ++writeBlob;
                }
                if (writeByte != yStart && runBytes > 0)
                    ValueStore::moveBytes(writeByte, yStart, runBytes);
                writeByte += runBytes;
                ++writeKey;
            }
            const int32_t removed = n - writeKey;
            if (removed != 0) {
                ks_.truncate(writeKey);
                ValueStore::truncateRunStarts(writeKey);
                ValueStore::truncateBlobStarts(writeBlob);
                ValueStore::truncatePool(writeByte);
                rebuildIndex();
            }
            return removed;
        }

        /// @brief Dump the dense blob-start column from `fromRow` — the
        ///        blob-starts tag (blob map).
        ///
        /// @param out     Byte sink.
        /// @param fromRow First blob index.
        void appendBlobStartBytes(std::vector<char>& out, int32_t fromRow) const {
            ValueStore::appendBlobStartBytes(out, fromRow);
        }

        /// @brief Bulk-append blob-start offsets — the blob-starts reload path.
        ///
        /// @param bytes    Source stream of `rowCount * 4` bytes.
        /// @param rowCount Blob starts to append; >= 0.
        void bulkLoadBlobStartBytes(const char* bytes, int64_t rowCount) {
            ValueStore::bulkLoadBlobStartBytes(bytes, rowCount);
        }

        /// @brief Dump the dense blob-pool bytes from byte offset `fromByte` —
        ///        the blob-pool tag (blob map).
        ///
        /// @param out      Byte sink.
        /// @param fromByte First pool byte offset.
        void appendBlobPoolBytes(std::vector<char>& out, int32_t fromByte) const {
            ValueStore::appendBlobPoolBytes(out, fromByte);
        }

        /// @brief Bulk-append blob-pool bytes — the blob-pool reload path.
        ///
        /// @param bytes    Source stream of `rowCount` bytes.
        /// @param rowCount Bytes to append; >= 0.
        void bulkLoadBlobPoolBytes(const char* bytes, int64_t rowCount) {
            ValueStore::bulkLoadBlobPoolBytes(bytes, rowCount);
        }

        // ---- Value-column deload (SingleValueStore / CsrValueStore) --------

        /// @brief Dump the dense value column from `fromRow` — the value tag.
        ///
        /// @param out     Byte sink.
        /// @param fromRow First value index.
        void appendValueBytes(std::vector<char>& out, int32_t fromRow) const {
            ValueStore::appendValueBytes(out, fromRow);
        }

        /// @brief Bulk-append values from a dense byte stream — value reload.
        ///
        /// @param bytes    Source stream of `rowCount * sizeof(V)` bytes.
        /// @param rowCount Values to append; >= 0.
        void bulkLoadValueBytes(const char* bytes, int64_t rowCount) {
            ValueStore::bulkLoadValueBytes(bytes, rowCount);
        }

        /// @brief Dump the dense run-start column from `fromRow` — the runs tag.
        ///
        /// @param out     Byte sink.
        /// @param fromRow First key index.
        void appendRunStartBytes(std::vector<char>& out, int32_t fromRow) const {
            ValueStore::appendRunStartBytes(out, fromRow);
        }

        /// @brief Bulk-append run-start offsets — the runs reload path.
        ///
        /// @param bytes    Source stream of `rowCount * 4` bytes.
        /// @param rowCount Run starts to append; >= 0.
        void bulkLoadRunStartBytes(const char* bytes, int64_t rowCount) {
            ValueStore::bulkLoadRunStartBytes(bytes, rowCount);
        }

        /// @brief Cross-arena deep copy — the LB-clone path.
        ///
        /// @details
        /// Re-appends `other`'s keys in id order into THIS set's arena
        /// (reproducing ids 1..count() exactly) and rebuilds the hash index.
        /// Asserts this set is empty.
        ///
        /// @param other Source set; resident.
        void copyFrom(const HashMap& other) {
            ks_.copyKeysFrom(other.ks_);
            ValueStore::copyValuesFrom(other);
            rebuildIndex();
        }

        /// @brief Wholesale reset to a fresh empty set — the `destroyGrid` path.
        ///
        /// @details
        /// Clears the cold key columns (escalating the aggregate dirty flag) and
        /// the throw-away hash buckets. Every previously minted id is invalid
        /// afterwards.
        void resetToFresh() {
            ks_.clear();
            buckets_.clear();
            ValueStore::clearValues();
            ++insertEpoch_;
            // Consistency: a fresh container is fully empty.
            assert(ks_.count() == 0 && buckets_.empty()
                && "cold-index desync: resetToFresh left state non-empty");
        }

        /// @brief Report the physical blocks held by this map's owning arena.
        ///
        /// @details
        /// This is arena-level telemetry, not a per-container allocation count:
        /// when several containers share one `LbArena`, each reports the same
        /// aggregate. It is exact for an exclusive arena such as the global
        /// `mailInterner` arena and includes spilled arena-directory blocks.
        /// The value is never a proof input.
        ///
        /// @return Blocks currently granted to the owning arena.
        /// @invariant The arena pointer supplied at construction remains valid
        ///            for the map's lifetime.
        /// @see LbArena::blocksHeld.
        int64_t arenaBlocksHeld() const {
            assert(arena_ != nullptr);
            return arena_->blocksHeld();
        }

        /// @brief Drop everything including the hash buckets' pages — the deload
        ///        release path.
        void release() {
            ks_.release();
            buckets_.clear();
            ValueStore::releaseValues();
        }

        /// @brief Rebuild the derived hash index from the stored keys — the
        ///        post-reload / `copyFrom` hook and the growth path of
        ///        `indexInsert`.
        ///
        /// @details
        /// Sizes the bucket array to a power of two above 2× the key count (1/2
        /// load), zeroes it (`reset`), and re-places every id. The transient
        /// scratch-free rebuild a throw-away container affords.
        void rebuildIndex() {
            const int32_t n = ks_.count();
            if (n == 0) {
                buckets_.clear();
                return;
            }
            int32_t cap = 64;
            while (cap < n * 2 + 2) cap <<= 1;
            buckets_.reset(cap);
            for (int32_t id = 1; id <= n; ++id) indexPlace(id);
            // Consistency: after a rebuild the index holds exactly the key
            // store's id set; a miss pins a desync to the reload / copyFrom /
            // erase / growth caller (G-54).
            assert(occupiedBuckets() == n
                && "cold-index desync: rebuildIndex occupied != key count");
        }

        // ---- Legacy ColdStringTable surface --------------------------------
        // These names + the LengthsView / BytesView facets below preserve the
        // hand-rolled byte-key interner's API so the seven LbMemory instances,
        // NameMap, and their tests compile unchanged through the
        // `using ColdStringTable = ColdHashSet<BytesKeyStore>` alias. Each
        // forwards to a BytesKeyStore-only method, so it INSTANTIATES ONLY for
        // the byte-key alias (a POD-key set never references them).

        /// @brief Legacy alias of `mint` (byte-key interner encode).
        ///
        /// @param s Bytes to intern.
        /// @return The string's id; >= 1.
        int32_t intern(const StrSpan& s) { return mint(s); }

        /// @brief Legacy alias of `decode` — a contiguous view of the cold
        ///        bytes.
        ///
        /// @param id A minted id; `1 <= id <= count()`.
        /// @return Span over the string's bytes.
        StrSpan view(int32_t id) const { return ks_.decodeAt(id); }

        /// @brief Legacy decode to an owned heap copy (page-aware).
        ///
        /// @param id A minted id; `1 <= id <= count()`.
        /// @return Heap copy of the bytes.
        std::string decodeString(int32_t id) const {
            return ks_.decodeStringAt(id);
        }

        /// @brief Legacy alias of `rebuildIndex`.
        void rebuildFindIndex() { rebuildIndex(); }

        /// @brief Lengths-column dump (deload facet helper).
        ///
        /// @param out     Byte sink.
        /// @param fromRow First string index.
        void appendLengthBytes(std::vector<char>& out, int32_t fromRow) const {
            ks_.appendLengthBytes(out, fromRow);
        }

        /// @brief Content-bytes dump (deload facet helper).
        ///
        /// @param out     Byte sink.
        /// @param fromRow First string index.
        void appendContentBytes(std::vector<char>& out, int32_t fromRow) const {
            ks_.appendContentBytes(out, fromRow);
        }

        /// @brief Total content bytes from `fromRow` (the bytes tag count).
        ///
        /// @param fromRow First string index counted.
        /// @return Sum of the counted strings' lengths.
        int64_t contentBytesFrom(int32_t fromRow) const {
            return ks_.contentBytesFrom(fromRow);
        }

        /// @brief Canonical reload from a lengths array + concatenated bytes,
        ///        rebuilding the heap index.
        ///
        /// @param lengths  `rowCount` int32 lengths in id order.
        /// @param bytes    The concatenated string bytes.
        /// @param byteLen  Total length of `bytes`; must equal the lengths' sum.
        /// @param rowCount Number of strings to append.
        void bulkLoad(const int32_t* lengths, const char* bytes,
                      int64_t byteLen, int32_t rowCount) {
            ks_.bulkLoadKeys(lengths, bytes, byteLen, rowCount);
            rebuildIndex();
            ++insertEpoch_;
        }

        /// @brief Key-set generation counter: bumped by every operation that
        ///        can ADD a key (`mint` on a miss, `resetToFresh`, `bulkLoad`).
        ///
        /// @details
        /// Erasures and value writes leave it unchanged. A consumer that
        /// derives a transient index over the key set (the equi-class hooks'
        /// `RejectedValidityBuckets`) records the epoch at build time and asserts
        /// it unchanged at every later use: a key the index has never seen cannot
        /// exist while the epoch stands, so a stale-entry filter (`lookup == 0`)
        /// is the only tolerance the index needs. Process-lifetime monotonic,
        /// never dumped, never a proof input.
        ///
        /// @return The current key-set epoch.
        uint32_t insertEpoch() const { return insertEpoch_; }

        /// @brief Stage one file set's lengths column (reload handshake).
        ///
        /// @param bytes    `rowCount * 4` bytes of int32 lengths.
        /// @param rowCount Number of strings in this set's window.
        void stageLengths(const char* bytes, int64_t rowCount) {
            ks_.stageLengths(bytes, rowCount);
        }

        /// @brief Consume the staged lengths with this set's content bytes,
        ///        rebuilding the heap index (reload handshake second half).
        ///
        /// @param bytes     The concatenated string bytes.
        /// @param byteCount Total content bytes.
        void consumeStagedLengths(const char* bytes, int64_t byteCount) {
            ks_.consumeStagedLengthsKeys(bytes, byteCount);
            rebuildIndex();
        }

        /// @brief Content dump from a BYTE offset (bytes tag tail window).
        ///
        /// @param out      Byte sink.
        /// @param fromByte Byte offset of the window start.
        void appendContentBytesFromByte(std::vector<char>& out,
                                        int64_t fromByte) const {
            ks_.appendContentBytesFromByte(out, fromByte);
        }

        /// @brief Serializer facet for the LENGTHS tag: one int32 per string,
        ///        id order.
        ///
        /// @details
        /// `LengthsView` + `BytesView` present a byte-key set to the generic
        /// deload visitor as its two GLDL tags. The lengths facet OWNS the
        /// whole-set lifecycle forwards (clear / release) — the bytes facet's are
        /// defined no-ops so the uniform visitor sweeps act exactly once per set.
        class LengthsView {
        public:
            using value_type = int32_t;

            /// @brief Facet over `owner` (stable address — an `LbMemory`
            ///        member).
            ///
            /// @param owner The owning set.
            explicit LengthsView(HashMap* owner) : owner_(owner) {
                assert(owner != nullptr);
            }

            /// @brief Bytes of the owning container's DERIVED key index.
            ///
            /// @details
            /// Present on the FIRST facet of a container only, so a caller
            /// walking every facet adds the index exactly once. Telemetry for
            /// the memory measurement; the index is never deloaded (I-117) and
            /// therefore has no `ContainerTag` of its own.
            ///
            /// @return Live bytes of the owner's hash index.
            int64_t indexBytes() const { return owner_->indexBytes(); }


            /// @brief String count (the lengths tag's element count).
            ///
            /// @return `count()` of the set.
            int32_t size() const { return owner_->count(); }

            /// @brief Dump the lengths column from `fromRow` (string index).
            ///
            /// @param out     Byte sink.
            /// @param fromRow First string index.
            void appendSpanBytes(std::vector<char>& out, int32_t fromRow) const {
                owner_->appendLengthBytes(out, fromRow);
            }

            /// @brief Reload: stash this set's lengths for the bytes facet's
            ///        join.
            ///
            /// @param bytes    The lengths column.
            /// @param rowCount Strings in this window.
            void bulkAppendBytes(const char* bytes, int64_t rowCount) {
                owner_->stageLengths(bytes, rowCount);
            }

            /// @brief Base-image clear: reset the whole set (the bytes facet's
            ///        `clear` is a no-op).
            void clear() { owner_->resetToFresh(); }

            /// @brief Deload release: drop the whole set (the bytes facet's
            ///        `release` is a no-op).
            void release() { owner_->release(); }

        private:
            HashMap* owner_;
        };

        /// @brief Serializer facet for the BYTES tag: the concatenated string
        ///        content, id order.
        class BytesView {
        public:
            using value_type = char;

            /// @brief Facet over `owner` (stable address — an `LbMemory`
            ///        member).
            ///
            /// @param owner The owning set.
            explicit BytesView(HashMap* owner) : owner_(owner) {
                assert(owner != nullptr);
            }

            /// @brief Total content bytes (the bytes tag's element count).
            ///
            /// @return Sum of all string lengths; asserts the int32 range the
            ///         deload count bookkeeping carries.
            int32_t size() const {
                const int64_t total = owner_->contentBytesFrom(0);
                assert(total <= INT32_MAX
                    && "cold key content exceeds the int32 deload count range");
                return static_cast<int32_t>(total);
            }

            /// @brief Dump the content from a BYTE offset (tail windows index
            ///        this tag in bytes).
            ///
            /// @param out     Byte sink.
            /// @param fromRow Byte offset of the window start.
            void appendSpanBytes(std::vector<char>& out, int32_t fromRow) const {
                owner_->appendContentBytesFromByte(
                    out, static_cast<int64_t>(fromRow));
            }

            /// @brief Reload: join this set's content bytes with the staged
            ///        lengths into one `bulkLoad`.
            ///
            /// @param bytes    The concatenated string bytes.
            /// @param rowCount Total content bytes in this window.
            void bulkAppendBytes(const char* bytes, int64_t rowCount) {
                owner_->consumeStagedLengths(bytes, rowCount);
            }

            /// @brief Defined no-op — the lengths facet already reset the set
            ///        this sweep (and must not disturb the lengths it staged
            ///        moments ago).
            void clear() {}

            /// @brief Defined no-op — the lengths facet owns the release.
            void release() {}

        private:
            HashMap* owner_;
        };

        /// @brief Serializer facet for a POD-key store's single dense KEY tag.
        ///
        /// @details
        /// The POD-key analog of `LengthsView` / `BytesView`, but self-contained:
        /// a fixed-size key needs no length column, so the key column is ONE
        /// deload tag. `lb_memory.hpp` names it to expose a POD-key
        /// `ColdHashSet` / `ColdHashMap` to the generic deload visitor. It OWNS
        /// the whole-container lifecycle (`clear` / `release`); a map's
        /// `ValuesView` facet's are no-ops. `bulkAppendBytes` rebuilds the
        /// throw-away index after the keys land (the index is keyed on keys only),
        /// so the key tag must precede the value tag in `ContainerTag` order.
        /// POD-key only — `value_type` is the stored key type (`KeyView == K` for
        /// `PodKeyStore`); the byte-key form uses `LengthsView` / `BytesView`.
        class KeysView {
        public:
            using value_type = typename KeyStore::KeyView;

            /// @brief Facet over `owner` (stable address — an `LbMemory` member).
            ///
            /// @param owner The owning container.
            explicit KeysView(HashMap* owner) : owner_(owner) {
                assert(owner != nullptr);
            }

            /// @brief Bytes of the owning container's DERIVED key index.
            ///
            /// @details
            /// Present on the FIRST facet of a container only, so a caller
            /// walking every facet adds the index exactly once. Telemetry for
            /// the memory measurement; the index is never deloaded (I-117) and
            /// therefore has no `ContainerTag` of its own.
            ///
            /// @return Live bytes of the owner's hash index.
            int64_t indexBytes() const { return owner_->indexBytes(); }


            /// @brief Key count (the key tag's element count).
            ///
            /// @return `count()` of the container.
            int32_t size() const { return owner_->count(); }

            /// @brief Dump the dense key column from `fromRow`.
            ///
            /// @param out     Byte sink.
            /// @param fromRow First key index.
            void appendSpanBytes(std::vector<char>& out, int32_t fromRow) const {
                owner_->keyStore().appendKeyBytes(out, fromRow);
            }

            /// @brief Reload: append the keys and rebuild the throw-away index.
            ///
            /// @param bytes    Source stream of `rowCount * sizeof(K)` bytes.
            /// @param rowCount Keys to append; >= 0.
            void bulkAppendBytes(const char* bytes, int64_t rowCount) {
                owner_->keyStore().bulkLoadKeyBytes(bytes, rowCount);
                // Reload-only rebuild of the throw-away index — the sole
                // rebuildIndex path off the hot mint / growth path, so the
                // timer read never lands on a fast path.
                const auto rebuildStart = std::chrono::steady_clock::now();
                owner_->rebuildIndex();
                deloadStats().recordIndexRebuild(
                    owner_->count(),
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - rebuildStart)
                        .count());
            }

            /// @brief Base-image clear: reset the whole container (the value
            ///        facet's `clear` is a no-op).
            void clear() { owner_->resetToFresh(); }

            /// @brief Deload release: drop the whole container (the value facet's
            ///        `release` is a no-op).
            void release() { owner_->release(); }

        private:
            HashMap* owner_;
        };

        /// @brief Serializer facet for a single-value map's dense VALUE tag.
        ///
        /// @details
        /// One tag (`values_[id-1]`). The `KeysView` facet owns the container
        /// lifecycle, so this facet's `clear` / `release` are defined no-ops; it
        /// only streams / reloads the parallel value column. Instantiates only for
        /// a `SingleValueStore` map (a set has no `ValueType`).
        class ValuesView {
        public:
            using value_type = typename ValueStore::ValueType;

            /// @brief Facet over `owner` (stable address — an `LbMemory` member).
            ///
            /// @param owner The owning container.
            explicit ValuesView(HashMap* owner) : owner_(owner) {
                assert(owner != nullptr);
            }

            /// @brief Value count — equals the key count for a single-value map.
            ///
            /// @return `count()` of the container.
            int32_t size() const { return owner_->count(); }

            /// @brief Dump the dense value column from `fromRow`.
            ///
            /// @param out     Byte sink.
            /// @param fromRow First value index.
            void appendSpanBytes(std::vector<char>& out, int32_t fromRow) const {
                owner_->appendValueBytes(out, fromRow);
            }

            /// @brief Reload: append the values (the `KeysView` facet, visited
            ///        first, already loaded the keys + rebuilt the index).
            ///
            /// @param bytes    Source stream of `rowCount * sizeof(V)` bytes.
            /// @param rowCount Values to append; >= 0.
            void bulkAppendBytes(const char* bytes, int64_t rowCount) {
                owner_->bulkLoadValueBytes(bytes, rowCount);
            }

            /// @brief Defined no-op — the key facet owns the reset.
            void clear() {}

            /// @brief Defined no-op — the key facet owns the release.
            void release() {}

        private:
            HashMap* owner_;
        };

        /// @brief Serializer facet for a CSR map's RUN-START column tag (one
        ///        int32 offset per key, id order).
        ///
        /// @details
        /// A `ColdSetMap` / `ColdMultiMap` deloads as THREE tags — keys, then
        /// run-starts, then values. `RunStartsView` is the middle one. Like
        /// `ValuesView`, the `KeysView` facet (visited first) owns the
        /// whole-container lifecycle, so this facet's `clear` / `release` are
        /// defined no-ops; it only streams / reloads the parallel run-start
        /// column. Instantiates only for a CSR value store (`runStartRaw` etc.).
        /// The run-start tag must follow the key tag and precede the value tag
        /// in `ContainerTag` order (reload loads keys + rebuilds the index, then
        /// the two parallel columns).
        class RunStartsView {
        public:
            using value_type = int32_t;

            /// @brief Facet over `owner` (stable address — an `LbMemory`
            ///        member).
            ///
            /// @param owner The owning container.
            explicit RunStartsView(HashMap* owner) : owner_(owner) {
                assert(owner != nullptr);
            }

            /// @brief Run-start count — one per key.
            ///
            /// @return `count()` of the container.
            int32_t size() const { return owner_->count(); }

            /// @brief Dump the dense run-start column from `fromRow`.
            ///
            /// @param out     Byte sink.
            /// @param fromRow First key index.
            void appendSpanBytes(std::vector<char>& out, int32_t fromRow) const {
                owner_->appendRunStartBytes(out, fromRow);
            }

            /// @brief Reload: append the run-start offsets (the `KeysView` facet,
            ///        visited first, already loaded the keys + rebuilt the index).
            ///
            /// @param bytes    Source stream of `rowCount * 4` bytes.
            /// @param rowCount Run starts to append; >= 0.
            void bulkAppendBytes(const char* bytes, int64_t rowCount) {
                owner_->bulkLoadRunStartBytes(bytes, rowCount);
            }

            /// @brief Defined no-op — the key facet owns the reset.
            void clear() {}

            /// @brief Defined no-op — the key facet owns the release.
            void release() {}

        private:
            HashMap* owner_;
        };

        /// @brief Serializer facet for a CSR map's VALUE column tag — the run
        ///        values concatenated in key-id order.
        ///
        /// @details
        /// The `SingleValueStore` `ValuesView` is keyed to `count()` (one value
        /// per key); a CSR map has `valueCount()` values across all runs, so it
        /// needs this distinct facet. The `KeysView` facet owns the lifecycle;
        /// this one only streams / reloads the value column. Instantiates only
        /// for a CSR value store.
        class RunValuesView {
        public:
            using value_type = typename ValueStore::ValueType;

            /// @brief Facet over `owner` (stable address — an `LbMemory`
            ///        member).
            ///
            /// @param owner The owning container.
            explicit RunValuesView(HashMap* owner) : owner_(owner) {
                assert(owner != nullptr);
            }

            /// @brief Total value count across every run.
            ///
            /// @return `valueCount()` of the container.
            int32_t size() const { return owner_->valueCount(); }

            /// @brief Dump the dense value column from `fromRow` (a value index).
            ///
            /// @param out     Byte sink.
            /// @param fromRow First value index.
            void appendSpanBytes(std::vector<char>& out, int32_t fromRow) const {
                owner_->appendValueBytes(out, fromRow);
            }

            /// @brief Reload: append the values (keys + run-starts already
            ///        loaded by the earlier facets).
            ///
            /// @param bytes    Source stream of `rowCount * sizeof(V)` bytes.
            /// @param rowCount Values to append; >= 0.
            void bulkAppendBytes(const char* bytes, int64_t rowCount) {
                owner_->bulkLoadValueBytes(bytes, rowCount);
            }

            /// @brief Defined no-op — the key facet owns the reset.
            void clear() {}

            /// @brief Defined no-op — the key facet owns the release.
            void release() {}

        private:
            HashMap* owner_;
        };

        /// @brief Serializer facet for a blob map's BLOB-START column tag — one
        ///        int32 byte offset per blob, blob-index order.
        ///
        /// @details
        /// A `ColdBlobMap` deloads as FOUR tags — keys, then run-starts, then
        /// blob-starts, then blob-pool. `BlobStartsView` is the third. Like the
        /// other CSR facets the `KeysView` (visited first) owns the lifecycle, so
        /// this facet's `clear` / `release` are no-ops; it only streams / reloads
        /// the parallel blob-start column. Instantiates only for a blob value
        /// store.
        class BlobStartsView {
        public:
            using value_type = int32_t;

            /// @brief Facet over `owner` (stable address — an `LbMemory` member).
            ///
            /// @param owner The owning container.
            explicit BlobStartsView(HashMap* owner) : owner_(owner) {
                assert(owner != nullptr);
            }

            /// @brief Blob count — one blob-start per blob.
            ///
            /// @return `blobCount()` of the container.
            int32_t size() const { return owner_->blobCount(); }

            /// @brief Dump the dense blob-start column from `fromRow`.
            ///
            /// @param out     Byte sink.
            /// @param fromRow First blob index.
            void appendSpanBytes(std::vector<char>& out, int32_t fromRow) const {
                owner_->appendBlobStartBytes(out, fromRow);
            }

            /// @brief Reload: append the blob-start offsets (keys + run-starts
            ///        already loaded by the earlier facets).
            ///
            /// @param bytes    Source stream of `rowCount * 4` bytes.
            /// @param rowCount Blob starts to append; >= 0.
            void bulkAppendBytes(const char* bytes, int64_t rowCount) {
                owner_->bulkLoadBlobStartBytes(bytes, rowCount);
            }

            /// @brief Defined no-op — the key facet owns the reset.
            void clear() {}

            /// @brief Defined no-op — the key facet owns the release.
            void release() {}

        private:
            HashMap* owner_;
        };

        /// @brief Serializer facet for a blob map's BLOB-POOL tag — the blob
        ///        bytes concatenated in blob-index order.
        ///
        /// @details
        /// The fourth and last `ColdBlobMap` tag. Dense (no no-straddle padding),
        /// so it streams / reloads with `appendSpanBytes` / `bulkAppendBytes`
        /// directly — `value_type = char`, `size()` the pool byte count, and the
        /// tail-window offset is a byte index (like `BytesView`). The `KeysView`
        /// facet owns the lifecycle; this one only moves bytes.
        class BlobPoolView {
        public:
            using value_type = char;

            /// @brief Facet over `owner` (stable address — an `LbMemory` member).
            ///
            /// @param owner The owning container.
            explicit BlobPoolView(HashMap* owner) : owner_(owner) {
                assert(owner != nullptr);
            }

            /// @brief Total pool bytes (the blob-pool tag's element count).
            ///
            /// @return `poolByteCount()` of the container.
            int32_t size() const { return owner_->poolByteCount(); }

            /// @brief Dump the pool bytes from byte offset `fromRow`.
            ///
            /// @param out     Byte sink.
            /// @param fromRow Byte offset of the window start.
            void appendSpanBytes(std::vector<char>& out, int32_t fromRow) const {
                owner_->appendBlobPoolBytes(out, fromRow);
            }

            /// @brief Reload: append the pool bytes (keys + run-starts +
            ///        blob-starts already loaded by the earlier facets).
            ///
            /// @param bytes    Source stream of `rowCount` bytes.
            /// @param rowCount Bytes to append; >= 0.
            void bulkAppendBytes(const char* bytes, int64_t rowCount) {
                owner_->bulkLoadBlobPoolBytes(bytes, rowCount);
            }

            /// @brief Defined no-op — the key facet owns the reset.
            void clear() {}

            /// @brief Defined no-op — the key facet owns the release.
            void release() {}

        private:
            HashMap* owner_;
        };

    private:
        /// @brief Occupied (non-zero) index slots; O(capacity). Pairs with
        ///        `ks_.count()` in the index/key-store consistency asserts
        ///        (G-54).
        int32_t occupiedBuckets() const {
            const int32_t cap = static_cast<int32_t>(buckets_.capacity());
            int32_t occ = 0;
            for (int32_t b = 0; b < cap; ++b)
                if (buckets_.at(b) != 0) ++occ;
            return occ;
        }

        /// @brief Insert `id` into the hash index, growing (via a rebuild) when
        ///        the load factor would reach 1/2.
        ///
        /// @param id The freshly minted id (already in the key store).
        void indexInsert(int32_t id) {
            // Consistency: mint always inserts the latest id.
            assert(id == ks_.count()
                && "cold-index desync: indexInsert id != latest key count");
            const int64_t n = ks_.count();
            if (buckets_.empty()
                || n * 2 >= static_cast<int64_t>(buckets_.capacity())) {
                rebuildIndex();     // sizes for n and places every id
                return;
            }
            indexPlace(id);
        }

        /// @brief Raw linear-probe insert into the bucket array (an IN-PLACE slot
        ///        write); no growth, no count update.
        ///
        /// @param id The id to place; its key must be resident.
        void indexPlace(int32_t id) {
            const uint64_t mask = buckets_.capacity() - 1;
            uint64_t i = ks_.hashStored(id) & mask;
            while (buckets_.at(static_cast<int32_t>(i)) != 0) {
                assert(buckets_.at(static_cast<int32_t>(i)) != id
                    && "ColdHashSet::indexPlace: duplicate id");
                i = (i + 1) & mask;
            }
            buckets_.set(static_cast<int32_t>(i), id);
        }

        LbArena* arena_;        // for the residency asserts on mint / lookup
        KeyStore ks_;           // the cold key columns (data cold)

        // Throw-away paged hash index (the "index heap" goes cold-but-transient
        // on this branch): open-addressing slots holding ids (0 = empty), probed
        // by the key store's content hash + key compare. Arena pages (no malloc),
        // never deloaded, rebuilt by rebuildIndex on reload — in-place slot
        // writes are safe precisely because it is never in a deload image
        // (I-117, throw-away-paged-hash form).
        PagedHashIndex buckets_;
        uint32_t insertEpoch_ = 0;   // key-set generation (see insertEpoch())
    };

    /// @brief The cold hash SET — keys only (the interner shape), the value
    ///        store being `EmptyValueStore` (elided by empty-base optimization,
    ///        so the layout is the hand-rolled set's). `ColdStringTable`
    ///        (`cold_string_table.hpp`) is the byte-key instantiation the seven
    ///        `LbMemory` interners and `NameMap` use.
    template <typename KeyStore>
    using ColdHashSet = HashMap<KeyStore, EmptyValueStore>;

    /// @brief The cold hash MAP — an arbitrary key → ONE stored value
    ///        (`SingleValueStore<V>`); set-once `insert`, `nullptr`-on-miss
    ///        `find`. Not yet wired into `LbMemory` — built + unit-tested for the
    ///        future int-keyed-map migration (D-165).
    template <typename KeyStore, typename V>
    using ColdHashMap = HashMap<KeyStore, SingleValueStore<V>>;

    /// @brief The cold MULTI-map — an arbitrary key → an ordered run of values
    ///        (`CsrValueStore<V>`, compressed-sparse-row form); append-to-tail
    ///        only. Not yet wired into `LbMemory` — built + unit-tested for the
    ///        future migration (D-165).
    template <typename KeyStore, typename V>
    using ColdMultiMap = HashMap<KeyStore, CsrValueStore<V>>;

    /// @brief The cold SET-MAP — an arbitrary key → a SORTED-UNIQUE set of
    ///        values (`SetValueStore<V>`, CSR form): `insertSorted` (interior,
    ///        deduped), `assignSet` (whole-run replace), `setContains`,
    ///        `eraseSet` / `eraseSetIf` (per-key, run-aware). The Batch-2
    ///        substrate for the int-keyed set-valued maps (`intToBeProved`,
    ///        `intStatementLevelsMap`, `orBookkeeping`), the shape a
    ///        `ColdMultiMap` could not serve (append-to-tail, no erase, no
    ///        dedup). Distinct from `ColdMultiMap` by value-store policy so the
    ///        bag's `appendToTail` and the set's `insertSorted` never mix
    ///        (D-168, I-118).
    template <typename KeyStore, typename V>
    using ColdSetMap = HashMap<KeyStore, SetValueStore<V>>;

    /// @brief The cold BLOB-map — an arbitrary key → an ordered run of
    ///        VARIABLE-LENGTH byte BLOBS (`BlobCsrValueStore`), each blob one
    ///        record's canonical serialization: `assignRun` (whole-run replace),
    ///        `blobAt` (decode a blob), `eraseBlobIf` (per-key, run-aware). The
    ///        record value store the family lacked — the substrate for
    ///        `equivalenceClassesMap` (Batch 3) and the Batch-4 `HashMemory`
    ///        record-set values, each supplying its own call-site codec. The store
    ///        is record-agnostic (no `V` template parameter — the value is bytes),
    ///        so the single-value / set surfaces never instantiate on it
    ///        (D-169, I-98).
    template <typename KeyStore>
    using ColdBlobMap = HashMap<KeyStore, BlobCsrValueStore>;

}
