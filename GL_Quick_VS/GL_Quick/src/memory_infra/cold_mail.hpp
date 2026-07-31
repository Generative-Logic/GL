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

// ColdMail -- the per-LB internal-mail mailbox
// (Memory::sameIterationInternalMail / nextIterationInternalMail), cold sibling
// of RoutingColdMail. The richer rationale lives on `struct ColdMail` below.

#include "dirty_state.hpp"
#include "lb_arena.hpp"
#include "mail_types.hpp"
#include "typed_cold_map.hpp"

#include <algorithm>
#include <cstdint>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace gl {

    /// @brief Reserved deload tag-block base for `LbMemory::sameInternalMail`.
    /// @details 10 facets at `base+0..9`; a 50-tag block is reserved (like the
    ///          HashMemory bands), placed after `exprOriginMap` (451-454).
    constexpr uint32_t kSameInternalMailDeloadBase = 455u;
    /// @brief Reserved deload tag-block base for `LbMemory::nextInternalMail`.
    constexpr uint32_t kNextInternalMailDeloadBase = 505u;

    /// @brief The per-LB internal-mail mailbox on the deloadable arena —
    ///        cold sibling of `RoutingColdMail`.
    ///
    /// @details
    /// The two internal-mail channels carry the cross-cycle revival traffic.
    /// Like the routing mailboxes (`RoutingColdMail`), the internal channels
    /// ride the deloadable arena; the internal channels additionally survive
    /// across bursts: `nextIterationInternalMail` carries one-step-delayed
    /// emissions
    /// across the deload seam, and `sameIterationInternalMail` carries the
    /// end-of-burst `sanitizeHashMemory` rewrites (written after the post-burst
    /// clear) to the next step. So the columns live on the LB's DELOADABLE
    /// per-LB `LbArena` (`LbMemory::manager`, passed in — NOT owned) with the
    /// LB's REAL deload-`dirty`, and are enumerated by `visitContainersImpl` at
    /// a reserved tag base so `LbMemory::visitContainers` deloads / reloads /
    /// releases / discharges them like any other cold container — the Batch-5
    /// `exprOriginMap` / `HashMemory`-at-a-base pattern that `RoutingColdMail`
    /// shares.
    ///
    /// Three live columns, all id-form (per-LB interner ids, decoded back to
    /// strings only at the `memory.hpp` boundary): `statements_` keyed on the
    /// whole pair (`IntMailStatementKey` = NameMap ids + levels) so
    /// `std::set<std::pair<EWV, std::set<int>>>` multiplicity is exact;
    /// `origins_` as a per-key blob run of `IntMailOrigin` keyed by the packed
    /// `originInterner` pair; `disintegrationSignals_` as a packed-`NameMap`-pair
    /// `uint8_t` (the two firing-time bools packed). Difference from
    /// `RoutingColdMail`: `ColdMail` carries `disintegrationSignals` (the internal
    /// channel is its only producer) and DROPS `expandedImplications` (always
    /// empty for internal mail). The column types and codecs
    /// (`IntMailStatementKey`, `IntMailOrigin`) are reused from `mail_types.hpp`.
    ///
    /// The cold maps mint in insertion order, so every observable boundary
    /// (`sortedStatements`, `sortedOrigins`, `sortedDisintegrationSignals`, and
    /// `makeHeapMail` in `memory.hpp`) re-imposes the canonical
    /// `ExpressionWithValidity::operator<` order the heap containers had; the
    /// deload stream stays a pure function of the (deterministic) insertion
    /// order.
    ///
    /// @invariant Single-threaded access: a given LB is owned by one worker at a
    ///            time and every internal-mail writer runs at a single-threaded
    ///            seam (`applyFiringRecords`, the post-join `updateGlobal*`
    ///            drain, the post-burst `standardProcessing`), so the per-LB
    ///            arena is never touched concurrently. [I-102]
    /// @see RoutingColdMail (the routing-channel sibling), Mail.
    struct ColdMail {
        /// @brief `set<pair<EWV, set<int>>>` as a set of id-form byte keys
        ///        (`IntMailStatementKey`: NameMap ids + levels).
        TypedColdSet<IntMailStatementKey> statements_;
        TypedColdSet<IntMailStatementKey>::LengthsView statementsLengths;
        TypedColdSet<IntMailStatementKey>::BytesView statementsBytes;
        /// @brief `map<EWV, vector<OriginLine>>` as an id-keyed blob run
        ///        (key = packed originInterner pair; value = `IntMailOrigin`).
        TypedColdBlobMap<int64_t, IntMailOrigin> origins_;
        TypedColdBlobMap<int64_t, IntMailOrigin>::KeysView originsKeys;
        TypedColdBlobMap<int64_t, IntMailOrigin>::RunStartsView originsRunStarts;
        TypedColdBlobMap<int64_t, IntMailOrigin>::BlobStartsView originsBlobStarts;
        TypedColdBlobMap<int64_t, IntMailOrigin>::BlobPoolView originsBlobPool;
        /// @brief `map<EWV, DisintegrationFlags>` as an id-keyed packed int32
        ///        (key = packed NameMap pair; bit 0 = `doNotDisintegrate`,
        ///        bit 1 = `allowOrDisintegration`, bits 2+ = the firing's
        ///        witness-generation `iteration + 1` — 0 means "nothing
        ///        carried", decoding to iteration -1).
        TypedColdMap<int64_t, int32_t> disintegrationSignals_;
        TypedColdMap<int64_t, int32_t>::KeysView disintegrationSignalsKeys;
        TypedColdMap<int64_t, int32_t>::ValuesView disintegrationSignalsValues;

        /// @brief Bind the three columns + their deload facets to the LB's
        ///        deloadable arena and real deload-dirty flag.
        ///
        /// @details Like the routing mailbox `RoutingColdMail`, `ColdMail`
        /// borrows `LbMemory`'s cold `manager` arena and `dirty` so its mutations
        /// escalate the LB's deload-dirty and its bytes ride the LB's deload
        /// stream. The facet views bind to each column's `inner()` (the engine
        /// map) so `LbMemory::visitContainers` can stream them.
        ///
        /// @param arena The LB's deloadable per-LB `LbArena` (`LbMemory::manager`).
        /// @param dirty The LB's deload-dirty flag (`LbMemory::dirty`).
        ColdMail(LbArena* arena, DirtyState* dirty)
            : statements_(arena, dirty),
              statementsLengths(&statements_.inner()),
              statementsBytes(&statements_.inner()),
              origins_(arena, dirty),
              originsKeys(&origins_.inner()),
              originsRunStarts(&origins_.inner()),
              originsBlobStarts(&origins_.inner()),
              originsBlobPool(&origins_.inner()),
              disintegrationSignals_(arena, dirty),
              disintegrationSignalsKeys(&disintegrationSignals_.inner()),
              disintegrationSignalsValues(&disintegrationSignals_.inner()) {}
        ColdMail(const ColdMail&) = delete;
        ColdMail& operator=(const ColdMail&) = delete;

        /// @brief Pack the two firing-time bools plus the witness-generation
        ///        iteration into the stored value.
        ///
        /// @details Bits 0-1 carry the two bools; bits 2+ carry
        /// `iteration + 1`, so the all-zero value keeps meaning "nothing
        /// carried" (decoding to iteration -1, the non-firing default the
        /// absorb maps to generation 0). `iteration` is the firing's
        /// `max premise iteration + 1` stamp
        /// (D-233); -1 means the writer carried
        /// no generation.
        ///
        /// @param dnd       `doNotDisintegrate`.
        /// @param aod       `allowOrDisintegration`.
        /// @param iteration The witness-generation stamp; `>= -1`.
        /// @return The packed int32.
        static int32_t packSignals(bool dnd, bool aod, int32_t iteration) {
            assert(iteration >= -1 && "packSignals: iteration below -1");
            return static_cast<int32_t>((dnd ? 1 : 0) | (aod ? 2 : 0))
                | ((iteration + 1) << 2);
        }

        /// @brief Unpack the iteration half of a packed signal value.
        /// @param packed The stored int32.
        /// @return The iteration (`-1` when the writer carried none).
        static int32_t unpackIteration(int32_t packed) {
            return (packed >> 2) - 1;
        }

        /// @brief Insert one statement (id-form pair + levels) — set semantics.
        ///
        /// @details Builds an `IntMailStatementKey` from the interned ids and the
        /// ascending level list and mints it; minting dedups by the whole key, so
        /// two members differing only in levels stay distinct (multiplicity
        /// preserved). The caller interns `(original, validityName)` into the LB's
        /// `NameMap` (the `insertInternalStatement` wrapper in `memory.hpp`).
        ///
        /// @param originalId The expression's `NameMap` id.
        /// @param validityId The scope's `NameMap` id.
        /// @param levels     The statement's level set (ascending in the byte key).
        void insertStatement(int32_t originalId, int32_t validityId,
            const std::set<int>& levels) {
            IntMailStatementKey k;
            k.originalId = originalId;
            k.validityId = validityId;
            k.levels.assign(levels.begin(), levels.end());
            statements_.mint(k);
        }

        /// @brief Insert one statement from an ASCENDING-UNIQUE level RUN —
        ///        the contiguous-run twin of the `std::set<int>` overload.
        ///
        /// @details
        /// Byte-identical deposit to the set overload when the precondition
        /// holds:
        /// 1. `[levels, levels + levelCount)` ascending strictly-unique is
        ///    exactly what `std::set<int>` iteration yields, so the key bytes
        ///    `k.levels` are IDENTICAL to the set overload's
        ///    `assign(levels.begin(), levels.end())`.
        /// 2. Identical key bytes mean identical `Codec<IntMailStatementKey>`
        ///    encoding, hence identical mint — dedup verdict, id, and deload
        ///    facet bytes.
        /// 3. `levelCount == 0` equals the empty-set case byte-for-byte
        ///    (empty `k.levels`).
        /// 4. No other observable differs — same `statements_.mint`, same
        ///    id sequence for never-before-seen keys.
        /// Lets callers reading a level run straight off a sorted-unique CSR
        /// store (`intStatementLevelsMap` values) deposit without building a
        /// transient `std::set<int>`.
        ///
        /// @pre `[levels, levels + levelCount)` is sorted ascending with no
        ///      duplicates (asserted for shape; the ORDER obligation is the
        ///      caller's source contract — a sorted-unique store run).
        /// @param originalId The expression's `NameMap` id.
        /// @param validityId The scope's `NameMap` id.
        /// @param levels     Pointer to `levelCount` ascending-unique levels;
        ///                   may be null only when `levelCount` is 0.
        /// @param levelCount The level count (`>= 0`).
        /// @see insertStatement(int32_t, int32_t, const std::set<int>&) — the
        ///      byte-oracle overload.
        void insertStatement(int32_t originalId, int32_t validityId,
            const int* levels, int32_t levelCount) {
            assert(levelCount >= 0 && (levelCount == 0 || levels != nullptr));
            IntMailStatementKey k;
            k.originalId = originalId;
            k.validityId = validityId;
            k.levels.assign(levels, levels + levelCount);
            statements_.mint(k);
        }

        /// @brief Set one statement's firing-time disintegration signals — the
        ///        `disintegrationSignals[key] = {dnd, aod, iteration}` write
        ///        door (id key).
        ///
        /// @details On a hit the two BOOLS overwrite in place (last-write —
        /// the canonically sorted record apply makes the order
        /// partition-independent) while the ITERATION half min-merges: a fact
        /// derivable at a lower generation is that generation, and the
        /// minimum is order-free so the merged value is deterministic
        /// regardless of write order. A carried-nothing new write
        /// (`iteration == -1`, packed 0) never overrides a real generation,
        /// and vice versa a real generation replaces carried-nothing. On a
        /// miss: set-once insert. The caller packs the statement's
        /// `(originalId, validityId)` `NameMap` pair into `key` via
        /// `packOriginKey` (the `setInternalDisintegrationSignal` wrapper in
        /// `memory.hpp`). Absent ⇒ `{false, false, -1}` at the receiver.
        ///
        /// @param key       The packed `(originalId, validityId)` statement key.
        /// @param dnd       `doNotDisintegrate`.
        /// @param aod       `allowOrDisintegration`.
        /// @param iteration The firing's witness-generation stamp; -1 = none.
        void setDisintegrationSignal(int64_t key, bool dnd, bool aod,
                                     int32_t iteration) {
            const int32_t id = disintegrationSignals_.lookup(key);
            if (id != 0) {
                const int32_t oldIt =
                    unpackIteration(disintegrationSignals_.valueAt(id));
                // min over real generations; -1 (none) loses to any real one.
                const int32_t mergedIt =
                    (oldIt == -1) ? iteration
                    : (iteration == -1) ? oldIt
                    : (oldIt < iteration ? oldIt : iteration);
                disintegrationSignals_.upsert(key,
                    packSignals(dnd, aod, mergedIt));
                return;
            }
            disintegrationSignals_.upsert(key, packSignals(dnd, aod, iteration));
        }

        /// @brief Read one statement's firing-time disintegration signals — the
        ///        `disintegrationSignals.find(key)` read door (id key).
        ///
        /// @details A non-minting probe of the packed-byte column; an absent key
        /// returns `{false, false}` — the same default the heap-`Mail` absorb
        /// applied. The caller packs the statement's `NameMap` pair into `key` via
        /// `packOriginKey`. The probe is const and never mints, so it is safe to
        /// call once per drained statement.
        ///
        /// @param key The packed `(originalId, validityId)` statement key.
        /// @return `{doNotDisintegrate, allowOrDisintegration}`; `{false, false}`
        ///         when no signal was set for `key`.
        std::pair<bool, bool> getDisintegrationSignal(int64_t key) const {
            const int32_t id = disintegrationSignals_.lookup(key);
            if (id == 0) return std::make_pair(false, false);
            const int32_t packed = disintegrationSignals_.valueAt(id);
            return std::make_pair((packed & 1) != 0, (packed & 2) != 0);
        }

        /// @brief Read one statement's firing-time disintegration signals into two
        ///        out-param bools — the heap-free peer of the pair-returning door.
        ///
        /// @details Identical semantics to
        /// `getDisintegrationSignal(int64_t) const` — a non-minting probe of the
        /// packed-byte column, an absent key yielding both bits `false` (the same
        /// default the heap-`Mail` absorb applied) — but it writes the two bits
        /// into caller-supplied `bool&`s instead of returning a `std::pair<bool,
        /// bool>`. The statement drain in `ExpressionAnalyzer::standardProcessing`
        /// reads the signals through this overload so the drain body holds no
        /// by-value `std::pair` local (0% heap; the pair form is retained as the
        /// byte-oracle for the twin test and for the out-of-tree test callers). The
        /// caller packs the statement's `(originalId, validityId)` `NameMap` pair
        /// into `key` via `packOriginKey`.
        ///
        /// @param key The packed `(originalId, validityId)` statement key.
        /// @param dnd Out: set to `doNotDisintegrate` (`false` when `key` absent).
        /// @param aod Out: set to `allowOrDisintegration` (`false` when `key`
        ///            absent).
        /// @param iteration Out: the firing's witness-generation stamp (`-1`
        ///            when `key` absent or the writer carried none — the
        ///            absorb's `-1 → 0` mapping keeps such deposits at
        ///            generation 0).
        /// @see getDisintegrationSignal(int64_t) const — the pair-returning oracle.
        void getDisintegrationSignal(int64_t key, bool& dnd, bool& aod,
                                     int32_t& iteration) const {
            const int32_t id = disintegrationSignals_.lookup(key);
            if (id == 0) {
                dnd = false;
                aod = false;
                iteration = -1;
                return;
            }
            const int32_t packed = disintegrationSignals_.valueAt(id);
            dnd = (packed & 1) != 0;
            aod = (packed & 2) != 0;
            iteration = unpackIteration(packed);
        }

        /// @brief Whether the statements column is empty.
        /// @return `true` if no statement has been inserted.
        bool statementsEmpty() const { return statements_.count() == 0; }
        /// @brief Whether the origin column is empty.
        /// @return `true` if no origin line has been added.
        bool exprOriginMapEmpty() const { return origins_.count() == 0; }
        /// @brief Whether the disintegration-signals column is empty.
        /// @return `true` if no signal has been set.
        bool disintegrationSignalsEmpty() const {
            return disintegrationSignals_.count() == 0;
        }
        /// @brief Whether all three columns are empty.
        /// @return `true` if the mailbox carries nothing.
        bool empty() const {
            return statementsEmpty() && exprOriginMapEmpty()
                && disintegrationSignalsEmpty();
        }

        // The string-decoding snapshots (sortedStatements / sortedOrigins /
        // sortedDisintegrationSignals) moved to memory.hpp as the free helpers
        // decodeInternalMailStatements / decodeInternalMailOrigins /
        // decodeInternalMailDisintegrationSignals + makeHeapMail, which see the
        // LB's NameMap / originInterner to turn the stored ids back into strings.
        // ColdMail itself stores only ids and exposes the id-form columns
        // directly (statements_ / origins_ / disintegrationSignals_).

        /// @brief Empty all three columns (pages back to the free-list) — the
        ///        per-step internal-mail clear (the `clearMail` lambda's target).
        void clear() {
            statements_.resetToFresh();
            origins_.resetToFresh();
            disintegrationSignals_.resetToFresh();
        }

        /// @brief Drop all column page capacity — the grid-teardown release.
        ///
        /// @details Releases the three columns' pages. The arena itself is owned
        /// by `LbMemory` (`manager`), which `releaseAll`s it at teardown; this
        /// only returns the per-column page capacity, mirroring `HashMemory`'s
        /// `release()`.
        void release() {
            statements_.release();
            origins_.release();
            disintegrationSignals_.release();
        }

        /// @brief Erase the `statements` entries the predicate accepts — the
        ///        scope-targeted `wipeSubtree` sweep (the `filterMailStatements`
        ///        twin).
        ///
        /// @details Only `statements` is swept (matching the heap, which leaves
        /// `origins` / `disintegrationSignals` untouched on a subtree wipe). The
        /// cold `eraseIf` run-compacts and rebuilds the index.
        ///
        /// @tparam Pred A callable `bool(const IntMailStatementKey&)`.
        /// @param pred Accepts the keys to erase (typically `inClosed(scope)`).
        /// @return Number of statement keys removed.
        template <typename Pred>
        int32_t filterStatements(Pred pred) {
            return statements_.eraseIf(
                [&pred](const IntMailStatementKey& k) { return pred(k); });
        }

        /// @brief Erase the `origins` runs whose packed key the predicate
        ///        accepts — the disproof-cleanup sweep's origins twin of
        ///        @ref filterStatements.
        ///
        /// @details
        /// `wipeSubtree` deliberately leaves internal-mail origins in place
        /// (matching the retired heap filter), which is invisible for a
        /// success-path wipe but leaves permanent orphaned history rows when a
        /// DISPROVED goal's scopes are wiped — the paired statements are
        /// filtered, so the origins would never be absorbed and would sit in
        /// the (discharge-surviving) container forever. The disproof cleanup
        /// calls this with the same closed-scope predicate it hands
        /// @ref filterStatements. The cold `eraseBlobIf` run-compacts and
        /// rebuilds the index.
        ///
        /// @tparam Pred A callable `bool(int64_t)` over the packed
        ///              `(exprId, validityId)` origin key (validity in the low
        ///              32 bits).
        /// @param pred Accepts the origin keys to erase.
        /// @return Number of origin keys removed.
        /// @see filterStatements — the statements sweep this pairs with.
        template <typename Pred>
        int32_t filterOrigins(Pred pred) {
            return origins_.eraseBlobIf(
                [&pred](int64_t k) { return pred(k); });
        }

        /// @brief Approximate live byte footprint across the three columns.
        /// @return Live bytes the mailbox occupies on the arena.
        int64_t liveBytes() const {
            return statements_.liveBytes() + origins_.liveBytes()
                + disintegrationSignals_.liveBytes();
        }

        /// @brief Enumerate the deload facets at `base + 0..7` — the seam
        ///        `LbMemory::visitContainers` splices each `ColdMail` instance in.
        ///
        /// @details `statements` (byte-key set) emits 2 facets at `base+0..1`;
        /// `origins` (int64-key blob map) 4 at `base+2..5`;
        /// `disintegrationSignals` (int64-key single-value map) 2 at `base+6..7`.
        /// Ascending, key-facet-first per column so reload rebuilds each index
        /// before its parallel columns. (Was 10 facets when the columns were
        /// byte-keyed; the id-form POD keys drop the Lengths+Bytes pair for one
        /// Keys facet on `origins` and `disintegrationSignals`.)
        ///
        /// @tparam Self    `ColdMail` or `const ColdMail` (dump reads const).
        /// @tparam Visitor Callable `(uint32_t tag, facet&)`.
        /// @param self  The instance being enumerated.
        /// @param base  The instance's reserved tag-block base.
        /// @param visit The per-facet visitor.
        template <typename Self, typename Visitor>
        static void visitContainersImpl(Self& self, uint32_t base,
            Visitor&& visit) {
            visit(base + 0u, self.statementsLengths);
            visit(base + 1u, self.statementsBytes);
            visit(base + 2u, self.originsKeys);
            visit(base + 3u, self.originsRunStarts);
            visit(base + 4u, self.originsBlobStarts);
            visit(base + 5u, self.originsBlobPool);
            visit(base + 6u, self.disintegrationSignalsKeys);
            visit(base + 7u, self.disintegrationSignalsValues);
        }

        /// @brief Mutable facet enumeration (load / release / discharge walk).
        /// @tparam Visitor Callable `(uint32_t tag, facet&)`.
        /// @param base  The instance's reserved tag-block base.
        /// @param visit The per-facet visitor.
        template <typename Visitor>
        void visitContainers(uint32_t base, Visitor&& visit) {
            visitContainersImpl(*this, base, std::forward<Visitor>(visit));
        }

        /// @brief Const facet enumeration (the deload DUMP reads a const LB).
        /// @tparam Visitor Callable `(uint32_t tag, const facet&)`.
        /// @param base  The instance's reserved tag-block base.
        /// @param visit The per-facet visitor.
        template <typename Visitor>
        void visitContainers(uint32_t base, Visitor&& visit) const {
            visitContainersImpl(*this, base, std::forward<Visitor>(visit));
        }
    };

    // addMailOrigin(ColdMail&, ...) moved to memory.hpp as the free helper
    // addInternalMailOrigin(ColdMail&, ValueInterner&, EWV, OriginLine, int):
    // its D-49 cap-full preference compares OriginTag values (equality1 /
    // equality2), and OriginTag lives in memory.hpp (invisible here). The helper
    // encodes (ev, origin) into the LB's originInterner + packs the key, then
    // does the same RMW on cm.origins_ directly.

} // namespace gl
