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

// ReverseArgsIndex -- the derived reverse membership side-index for the
// per-LB secondary index `remainingArgsNormalizedEncodedMap`
// (`TypedColdBlobMap<Int16SetKey, NormKey>`, forward direction: an arg-id set
// key -> a run of the normalized-premise keys that carry exactly those args).
//
// The firing-check candidate loop (`checkLocalEncodedMemoryStatic`) needs the
// OPPOSITE direction: given the request's normalized key (a `NormKey`), the
// set of forward keys whose stored run CONTAINS that NormKey. Before this
// index that answer cost a full O(keys) scan of the forward map plus a
// per-candidate O(run) byte-peek membership probe. This container answers it
// as one hash probe + a walk of the owning-key chain.
//
// It is DERIVED (I-117 pattern, the cold-map family's key->id index): it lives
// on the LB's deloadable `LbArena` (pool-backed, zero heap) but is NEVER
// enrolled in `LbMemory::visitContainers`, NEVER in the canonical deload
// stream, and NEVER dumped -- it is rebuilt on canonical reload from the
// forward map's cold runs, exactly as `HashMap::rebuildIndex` rebuilds the
// throw-away `PagedHashIndex` buckets. The v4 RAW eviction image captures its
// arena pages verbatim (a whole-arena memcpy), so a raw round-trip restores it
// without a rebuild, again mirroring the cold-map buckets.

#include "cold_hash_map.hpp"
#include "paged_vector.hpp"
#include "scratch_arena.hpp"

#include <cassert>
#include <cstdint>

namespace gl {

    /// @brief Derived NormKey -> owning-forward-key reverse membership index for
    ///        `remainingArgsNormalizedEncodedMap`.
    ///
    /// @details
    /// The forward map keys an arg-id set (`Int16SetKey`) to a sorted-unique run
    /// of normalized premise keys (`NormKey` value blobs). This side-index
    /// inverts that relation: it maps the BYTES of a `NormKey`
    /// (`Codec<NormKey>::serialize`, which is byte-identical to the key
    /// `encode`) to the run of forward-map key ids whose stored run contains that
    /// exact NormKey.
    ///
    /// **Substrate (zero heap, derived, throw-away).** Three cooperating pieces,
    /// all on the owning LB's `LbArena` page/byte tier:
    /// - `normKeys_` — a `ColdHashSet<BytesKeyStore>` interning each DISTINCT
    ///   NormKey's bytes to a dense entry id (`1..count()`). This is the
    ///   `PagedHashIndex` (its internal `buckets_`) plus a byte pool; membership
    ///   is by EXACT bytes, so a hash collision between two distinct NormKeys is
    ///   resolved by the byte compare inside `BytesKeyStore` — the reverse
    ///   answer carries NO false positives.
    /// - `headById_` — a `PagedVector<int32_t>` indexed by `entryId - 1`, the
    ///   head node index of that NormKey's owner chain (`-1` = empty).
    /// - `nodeOwner_` / `nodeNext_` — two parallel `PagedVector<int32_t>` columns
    ///   forming the per-entry singly-linked owner chain (`nodeOwner_[n]` = a
    ///   forward-map key id; `nodeNext_[n]` = the next node or `-1`). The chain
    ///   encodes the per-NormKey "run of owning ids" the design calls for;
    ///   chaining is chosen over a strict CSR value column precisely so
    ///   @ref appendEdge is O(1) — a CSR interior splice would reintroduce the
    ///   O(tail) memmove this whole campaign removes.
    ///
    /// **Lifecycle (mirrors the cold-map buckets exactly).**
    /// - Maintained incrementally by @ref appendEdge at the single-threaded
    ///   install sites (one edge per NormKey newly added to a forward key's run).
    /// - Rebuilt wholesale by @ref rebuildReverseIndex from the forward map's
    ///   cold runs — the canonical-reload hook and the post-wipe re-derivation.
    /// - Freed page-by-page by @ref clear at the canonical release seam BEFORE
    ///   the arena's blocks are reclaimed (the throw-away pages must not outlive
    ///   the arena as stale vids).
    /// - Read (const) during the parallel firing phase; every write is
    ///   single-threaded (I-83), so no concurrent-mutation hazard.
    ///
    /// It carries its OWN private `DirtyState` (@ref derivedDirty_) rather than
    /// the LB's shared flag: a derived container is never serialized, so its
    /// mutations must not mark the LB dirty (that would force a needless deload
    /// dump of an otherwise-unchanged LB).
    ///
    /// @invariant The set of owner ids @ref reverseIndexRunOf yields for a
    ///            NormKey equals `{ forward key id : that key's run contains the
    ///            NormKey }` EXACTLY — no false positives (exact-byte membership
    ///            in `normKeys_`), no omissions (every run member gets an edge),
    ///            no duplicates (the forward run is sorted-unique and
    ///            @ref appendEdge fires once per newly-added NormKey).
    /// @invariant Derived (I-117): never deloaded, never enrolled in
    ///            `LbMemory::visitContainers`, never dumped; rebuilt on canonical
    ///            reload, captured verbatim by the raw image.
    /// @see `PagedHashIndex`, `HashMap::rebuildIndex`, `ColdHashSet`,
    ///      `remainingArgsNormalizedEncodedMap`, `insertRemainingArgsNormKey`,
    ///      `wipeRemainingArgsForClosed`, D-199,
    ///      I-154.
    class ReverseArgsIndex {
    public:
        /// @brief Bind to the owning LB's arena (the page/byte source).
        ///
        /// @details
        /// Lazy like every arena container: construction touches neither the
        /// arena nor a pool block, so an unused index consumes nothing. All
        /// three storage columns and the interner bind the SAME arena and the
        /// index's OWN private dirty flag.
        ///
        /// @param arena The LB's arena. Must be non-null.
        explicit ReverseArgsIndex(LbArena* arena)
            : arena_(arena),
              normKeys_(arena, &derivedDirty_),
              headById_(arena, &derivedDirty_),
              nodeOwner_(arena, &derivedDirty_),
              nodeNext_(arena, &derivedDirty_) {
            assert(arena_ != nullptr);
        }

        ReverseArgsIndex(const ReverseArgsIndex&) = delete;
        ReverseArgsIndex& operator=(const ReverseArgsIndex&) = delete;
        ReverseArgsIndex(ReverseArgsIndex&&) = delete;
        ReverseArgsIndex& operator=(ReverseArgsIndex&&) = delete;

        /// @brief Record one reverse edge: the NormKey @p normKeyBytes is carried
        ///        by the forward-map key id @p ownerKeyId.
        ///
        /// @details
        /// Interns @p normKeyBytes into `normKeys_` (find-or-mint, so a repeated
        /// NormKey re-uses its entry id), grows `headById_` to cover a freshly
        /// minted entry, then PREPENDS a new owner node onto that NormKey's chain
        /// — O(1), no CSR tail shift. The caller guarantees `(normKey,
        /// ownerKeyId)` is not already present (`insertRemainingArgsNormKey`
        /// appends the edge only when the NormKey was newly added to that key's
        /// sorted-unique run), so the chain stays duplicate-free without a scan.
        ///
        /// Single-threaded install only (I-83): the parallel firing phase reads
        /// this index but never mutates it.
        ///
        /// @param normKeyBytes The NormKey's serialized bytes
        ///                     (`Codec<NormKey>::serialize` == key `encode`).
        /// @param ownerKeyId   The forward-map key id (`1..forward.count()`) whose
        ///                     run now contains @p normKeyBytes; asserted `>= 1`.
        /// @invariant The prepended node's owner id is unique in the chain (the
        ///            caller's newly-added contract); asserted only implicitly via
        ///            the exact-set @ref reverseIndexRunOf invariant.
        /// @see `insertRemainingArgsNormKey`, @ref rebuildReverseIndex.
        void appendEdge(StrSpan normKeyBytes, int32_t ownerKeyId) {
            assert(ownerKeyId >= 1
                && "ReverseArgsIndex::appendEdge: owner key id must be >= 1");
            const int32_t entryId = normKeys_.mint(normKeyBytes);
            assert(entryId >= 1
                && "ReverseArgsIndex::appendEdge: mint returned a null id");
            while (headById_.size() < entryId) headById_.push_back(-1);
            const int32_t oldHead = headById_[entryId - 1];
            const int32_t newNode = nodeOwner_.size();
            nodeOwner_.push_back(ownerKeyId);
            nodeNext_.push_back(oldHead);
            headById_.setAt(entryId - 1, newNode);
        }

        /// @brief Enumerate the forward-map key ids whose run contains the NormKey
        ///        @p normKeyBytes — the firing-check candidate probe.
        ///
        /// @details
        /// One `normKeys_.lookup` (exact-byte, hash-collision-safe) yields the
        /// NormKey's entry id (or 0 = never recorded, a DEFINED miss, not a
        /// fallback), then walks that entry's owner chain, invoking @p sink once
        /// per owning forward-map key id. The yielded set is exactly the run-
        /// contains membership set (the class invariant) with no duplicates, so
        /// the caller may collect the ids directly with no dedup. Chain order is
        /// append/rebuild order and is NOT observable: the candidate loop re-sorts
        /// the collected ids by `int16SetKeyLexCompare` before emission.
        ///
        /// @tparam Sink Callable `void(int32_t ownerKeyId)`.
        /// @param normKeyBytes The probe NormKey's serialized bytes.
        /// @param sink         Invoked once per owning forward-map key id.
        /// @see @ref appendEdge, `checkLocalEncodedMemoryStatic`.
        template <typename Sink>
        void reverseIndexRunOf(StrSpan normKeyBytes, Sink&& sink) const {
            const int32_t entryId = normKeys_.lookup(normKeyBytes);
            if (entryId == 0) return;
            assert(entryId >= 1 && entryId <= headById_.size()
                && "ReverseArgsIndex::reverseIndexRunOf: entry id out of range");
            for (int32_t node = headById_[entryId - 1]; node != -1;
                 node = nodeNext_[node]) {
                sink(nodeOwner_[node]);
            }
        }

        /// @brief Rebuild the whole index from the forward map's cold runs — the
        ///        canonical-reload hook and the post-wipe re-derivation.
        ///
        /// @details
        /// Frees any current pages (@ref clear), then walks every forward key id
        /// `1..forward.count()` and every blob `0..runLen(id)` in that key's run,
        /// appending a reverse edge `(blob bytes -> id)` via @ref appendEdge. The
        /// blob bytes ARE the NormKey's serialized bytes (`Codec<NormKey>`), read
        /// zero-decode through `peekRecordBytes`. O(sum of run lengths) with the
        /// interner's amortized-O(1) mint — the "rebuild dense from live content"
        /// the derived-index substrate already runs at reload.
        ///
        /// Requires a resident arena (@ref clear frees pages; `peekRecordBytes`
        /// asserts residency). The result is independent of forward-key
        /// iteration order and of chain order, so it is byte-invisible to every
        /// observable (I-107 / I-117).
        ///
        /// @tparam ForwardMap A `TypedColdBlobMap`-shaped map exposing
        ///                    `count()`, `runLen(id)`, and
        ///                    `peekRecordBytes(id, j, len, scratch)`.
        /// @param forward The forward secondary index to invert (resident).
        /// @param scratch A per-slot scratch arena for `peekRecordBytes` page
        ///                straddles (cursor/popTo framed by the caller).
        /// @see @ref appendEdge, `HashMap::rebuildIndex`,
        ///      `wipeRemainingArgsForClosed`.
        template <typename ForwardMap>
        void rebuildReverseIndex(const ForwardMap& forward,
                                 ScratchArena& scratch) {
            clear();
            const int32_t n = forward.count();
            for (int32_t id = 1; id <= n; ++id) {
                const int32_t rl = forward.runLen(id);
                for (int32_t j = 0; j < rl; ++j) {
                    int32_t bl = 0;
                    const char* bp = forward.peekRecordBytes(id, j, bl, scratch);
                    appendEdge(StrSpan(bp, bl), id);
                }
            }
        }

        /// @brief Free every held page back to the arena and reset to empty — the
        ///        canonical release seam and the head of @ref rebuildReverseIndex.
        ///
        /// @details
        /// A LIVE operation: `release()` / `clear()` on the arena containers
        /// require a resident arena, and their own residency asserts are the
        /// tripwire (Rule 19). A raw-deloaded LB's teardown never reaches here —
        /// the members' `~PagedVector` / `~ColdHashSet` (`~PagedHashIndex`)
        /// residency branches make destruction on a cold arena a no-op.
        void clear() {
            normKeys_.release();
            headById_.clear();
            nodeOwner_.clear();
            nodeNext_.clear();
            derivedDirty_ = DirtyState::Clean;
        }

        /// @brief Whether the index holds no recorded NormKeys.
        ///
        /// @return `true` when no NormKey has ever been recorded (or after
        ///         @ref clear).
        bool empty() const { return normKeys_.count() == 0; }

        /// @brief Distinct-NormKey count (entry ids in use).
        ///
        /// @return The number of distinct NormKeys recorded; `>= 0`.
        int32_t distinctKeyCount() const { return normKeys_.count(); }

        /// @brief Live arena-page footprint in bytes — the steward's pressure
        ///        input (the throw-away pages are real arena pages while
        ///        resident).
        ///
        /// @return Held bytes across the interner and the three columns.
        int64_t liveBytes() const {
            return normKeys_.liveBytes() + headById_.liveBytes()
                 + nodeOwner_.liveBytes() + nodeNext_.liveBytes();
        }

    private:
        /// @brief Private content-change flag — never read for deload (the index
        ///        is derived), kept only because the arena containers require a
        ///        `DirtyState*`. Declared FIRST so it outlives the containers
        ///        that reference it (destruction is reverse-declaration order).
        DirtyState derivedDirty_ = DirtyState::Clean;
        /// @brief The owning LB's arena (page/byte source).
        LbArena* arena_;
        /// @brief Distinct NormKey bytes -> dense entry id (`1..count()`); the
        ///        `PagedHashIndex` + byte pool that make membership exact.
        ColdHashSet<BytesKeyStore> normKeys_;
        /// @brief `entryId - 1` -> owner-chain head node (`-1` = empty).
        PagedVector<int32_t> headById_;
        /// @brief node -> owning forward-map key id.
        PagedVector<int32_t> nodeOwner_;
        /// @brief node -> next node in the same NormKey's chain (`-1` = tail).
        PagedVector<int32_t> nodeNext_;
    };

}
