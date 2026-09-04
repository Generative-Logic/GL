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

// RuleIndexStaging -- the per-absorb staging of a HashMemory's rule-index
// writes (D-333).
//
// A rule install writes, per multiplied copy and per surviving permutation,
// one record into each of seven cold blob maps (the LMV map, the whole-key /
// originals / copy-owner / remaining-args-owner runs, the subkey record map
// and the remaining-args forward map). Every one of those writes used to be
// an in-place splice into a dense CSR pool, O(bytes behind the key), and the
// same key received a record per permutation and per rule sharing it. The
// staging collects one absorb's records per map and per key; the flush
// (`ExpressionAnalyzer::flushRuleIndexStaging`) then rewrites each touched
// map ONCE, in key-id order, merging the staged records into every key's run
// exactly as the sequential doors would have (sorted-unique owner runs,
// arrival-ordered LMV runs, the merged subkey record, the sorted-unique
// remaining-args run), and mints the new keys in first-touch order -- so the
// cold bytes after the flush equal the bytes the sequential writes produce.
//
// Substrate: the stagings ride the WORKER SLOT's gen-scratch arena (page
// tier, zero heap), a fixed pool per slot, each bound to the hash-memory
// instance it currently serves (`RuleStagingRegistry::acquire`); nested
// installs across LBs hold one binding each until their window closes; a
// binding changes only at a window close (`closeRuleIndexStaging`: flush the
// LB's four instances, release). A staging window never crosses a task (the
// absorb and apply windows close inside their `standardProcessing`; an
// install whose exit closes its LB's outermost window closes all four
// instances, since its nested rounds may have staged into any of them), so
// every staging of the slot is unbound at every task end and every seam a
// reader reaches (asserted), and its pages return to the slot's free list at
// the flush -- no LB arena grows by a window's staging.

#include "cold_hash_map.hpp"
#include "dirty_state.hpp"
#include "lb_arena.hpp"
#include "paged_vector.hpp"
#include "scratch_arena.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <deque>

namespace gl {

    /// @brief One staged rule-index record: a node of its key's arrival-order
    ///        chain inside a `RuleIndexStagingSection`.
    ///
    /// @details
    /// A flat 24-byte POD (PagedVector element). Which fields carry meaning
    /// depends on the section: the owner runs use `owner` only; the LMV and
    /// remaining-args sections use the payload (`off` / `len`) only; the
    /// subkey section uses `owner` plus either a signature payload (the
    /// owner's u_ signature as `(int32 slot, int32 id)` pairs) or the loose
    /// marker `flag == 1` (a prefix without u_ arguments, no signature).
    struct StagedRuleItem {
        int32_t next;    ///< Next item of the same key in arrival order; -1 = last.
        int32_t off;     ///< Payload offset in the section's byte pool; -1 = none.
        int32_t len;     ///< Payload byte length; 0 = none.
        int32_t flag;    ///< Section-specific marker (subkeys: 1 = loose contribution).
        int64_t owner;   ///< The packed `RuleOwner`; 0 where the section stores none.
    };

    static_assert(sizeof(StagedRuleItem) == 24,
                  "StagedRuleItem must stay a flat 24-byte POD (PagedVector element)");

    /// @brief The staged records of ONE cold map: key bytes interned in
    ///        first-touch order, per key an arrival-ordered chain of items,
    ///        variable payloads in a byte pool.
    ///
    /// @details
    /// Three cooperating pieces, all on the owning LB's arena page tier:
    /// - `keys_` -- a `ColdHashSet<BytesKeyStore>` interning each distinct
    ///   key's bytes to a dense ordinal (`1..keyCount()`), in the order the
    ///   keys are first staged. The flush mints the keys the cold map lacks in
    ///   exactly this order, which is the order the sequential writes minted
    ///   them (the first write of a key is its first staging).
    /// - `head_` / `tail_` -- per ordinal the first and last item of the key's
    ///   chain (`-1` = empty); appending at the tail keeps arrival order, which
    ///   the LMV run (arrival-ordered) depends on.
    /// - `items_` / `bytes_` -- the chain nodes and the payload pool.
    ///
    /// The section carries its own `DirtyState`: it is never serialized, so
    /// its writes must not mark the LB dirty. `keyCount_` / `itemCount_` are
    /// plain integers, so `empty()` never touches the arena (safe on a
    /// deloaded LB -- the emptiness assert at the deload seam relies on it).
    /// Single-threaded write side (the install runs under the LB's worker
    /// claim, I-83); no reader during the parallel phases (the flush precedes
    /// them, asserted).
    ///
    /// @invariant `forEachItem(ord)` visits the key's items in the order they
    ///            were staged; `keyAt(ord)` for `ord` ascending is first-touch
    ///            order.
    /// @see `RuleIndexStaging`, `ExpressionAnalyzer::flushRuleIndexStaging`,
    ///      `ReverseArgsIndex` (the same interner-plus-chain substrate).
    class RuleIndexStagingSection {
    public:
        /// @brief Bind the section to the owning LB's arena, empty.
        ///
        /// @param arena The LB's deloadable arena; outlives the section.
        explicit RuleIndexStagingSection(LbArena* arena)
            : keys_(arena, &dirty_), head_(arena, &dirty_), tail_(arena, &dirty_),
              items_(arena, &dirty_), bytes_(arena, &dirty_) {
            assert(arena != nullptr);
        }

        RuleIndexStagingSection(const RuleIndexStagingSection&) = delete;
        RuleIndexStagingSection& operator=(const RuleIndexStagingSection&) = delete;

        /// @brief Number of distinct keys staged (arena-free probe).
        /// @return The key count.
        int32_t keyCount() const { return keyCount_; }

        /// @brief Number of items staged across every key (arena-free probe).
        /// @return The item count.
        int32_t itemCount() const { return itemCount_; }

        /// @brief Whether nothing is staged (arena-free probe).
        /// @return `true` when no item is staged.
        bool empty() const { return itemCount_ == 0; }

        /// @brief Stage one record under a key.
        ///
        /// @details
        /// Interns the key on its first appearance (its ordinal is the next
        /// one), copies the payload into the byte pool, and appends the item at
        /// the tail of the key's chain. The key bytes and the payload are
        /// copied, so the caller's buffers may die on return.
        ///
        /// @param keyBytes The cold map's key bytes (the map's `Codec<K>`
        ///                 encoding), `len > 0`.
        /// @param owner    The packed `RuleOwner`, or 0 where the section
        ///                 stores none.
        /// @param payload  The payload bytes; read only when `len > 0`.
        /// @param len      The payload length; `>= 0`.
        /// @param flag     The section-specific marker.
        /// @return The key's ordinal (`>= 1`).
        int32_t stage(StrSpan keyBytes, int64_t owner,
                      const char* payload, int32_t len, int32_t flag) {
            assert(keyBytes.len > 0 && "RuleIndexStagingSection::stage: empty key");
            assert(len >= 0 && (len == 0 || payload != nullptr)
                && "RuleIndexStagingSection::stage: payload length without bytes");
            const int32_t ord = keys_.mint(keyBytes);
            assert(ord >= 1 && ord <= keyCount_ + 1
                && "RuleIndexStagingSection::stage: interner ordinal out of sequence");
            if (ord == keyCount_ + 1) {
                head_.push_back(-1);
                tail_.push_back(-1);
                ++keyCount_;
            }
            StagedRuleItem it;
            it.next = -1;
            it.off = -1;
            it.len = len;
            it.flag = flag;
            it.owner = owner;
            if (len > 0) {
                // Page-contiguous: a payload never straddles a pool page, so
                // the flush's comparators read it in place (no scratch copy).
                it.off = bytes_.appendRunNoStraddle(payload, len);
            }
            const int32_t node = items_.size();
            items_.push_back(it);
            const int32_t last = tail_[ord - 1];
            if (last < 0) {
                head_.setAt(ord - 1, node);
            } else {
                StagedRuleItem prev = items_[last];
                assert(prev.next == -1
                    && "RuleIndexStagingSection::stage: chain tail is not the last item");
                prev.next = node;
                items_.setAt(last, prev);
            }
            tail_.setAt(ord - 1, node);
            ++itemCount_;
            return ord;
        }

        /// @brief The key bytes of ordinal @p ord (a view into the interner's
        ///        pool, valid until `clear`).
        /// @param ord A staged ordinal in `[1, keyCount()]`.
        /// @return The key bytes.
        StrSpan keyAt(int32_t ord) const {
            assert(ord >= 1 && ord <= keyCount_
                && "RuleIndexStagingSection::keyAt: ordinal out of range");
            return keys_.keyAt(ord);
        }

        /// @brief The ordinal of a key, `0` when it is not staged.
        /// @param keyBytes The key bytes.
        /// @return The ordinal or `0`.
        int32_t lookup(StrSpan keyBytes) const { return keys_.lookup(keyBytes); }

        /// @brief Visit the items of ordinal @p ord in arrival order.
        ///
        /// @tparam Fn Callable `void(const StagedRuleItem&)`.
        /// @param ord A staged ordinal in `[1, keyCount()]`.
        /// @param fn  The consumer.
        /// @return Nothing.
        template <typename Fn>
        void forEachItem(int32_t ord, Fn fn) const {
            assert(ord >= 1 && ord <= keyCount_
                && "RuleIndexStagingSection::forEachItem: ordinal out of range");
            for (int32_t n = head_[ord - 1]; n >= 0;) {
                const StagedRuleItem it = items_[n];
                fn(it);
                n = it.next;
            }
        }

        /// @brief In-place view of an item's payload -- a payload is stored
        ///        page-contiguous (`appendRunNoStraddle`), so the pointer spans
        ///        all `it.len` bytes and stays valid until `clear`.
        ///
        /// @param it A staged item of this section.
        /// @return A pointer to `it.len` contiguous bytes; `nullptr` when the
        ///         item has no payload.
        const char* payload(const StagedRuleItem& it) const {
            if (it.len == 0) return nullptr;
            assert(it.off >= 0 && it.off + it.len <= bytes_.size()
                && "RuleIndexStagingSection::payload: item outside the pool");
            int32_t run = 0;
            const char* p = bytes_.contiguousRun(it.off, run);
            assert(run >= it.len
                && "RuleIndexStagingSection::payload: a payload straddles a pool page");
            return p;
        }

        /// @brief Forget everything and return every page to the arena.
        /// @return Nothing.
        void clear() {
            keys_.release();
            head_.clear();
            tail_.clear();
            items_.clear();
            bytes_.clear();
            keyCount_ = 0;
            itemCount_ = 0;
            dirty_ = DirtyState::Clean;
        }

    private:
        DirtyState dirty_ = DirtyState::Clean;   // declared first: the containers take its address
        ColdHashSet<BytesKeyStore> keys_;
        PagedVector<int32_t> head_;
        PagedVector<int32_t> tail_;
        PagedVector<StagedRuleItem> items_;
        PagedVector<char> bytes_;
        int32_t keyCount_ = 0;
        int32_t itemCount_ = 0;
    };

    /// @brief The seven staging sections of one `HashMemory` -- one per rule
    ///        index map the install writes.
    ///
    /// @details
    /// Section to map: `wholeKeys` -> `normalizedEncodedKeys`, `originals` ->
    /// `originals`, `copyOwners` -> `copyOwners`, `remArgsOwners` ->
    /// `remainingArgsOwners` (the four sorted-unique owner runs); `lmv` ->
    /// `encodedMap` (arrival-ordered LMV blobs); `subkeys` ->
    /// `normalizedEncodedSubkeys` (one merged `OwnerSet` record per key);
    /// `remArgs` -> `remainingArgsNormalizedEncodedMap` (sorted-unique
    /// `NormKey` blobs; the flush also appends the reverse-index edges of the
    /// genuinely new ones).
    ///
    /// @see `RuleIndexStagingSection`, `RuleStagingRegistry`,
    ///      `ExpressionAnalyzer::flushRuleIndexStaging`.
    struct RuleIndexStaging {
        /// @brief Bind every section to an arena, empty.
        /// @param arena The arena (a worker slot's gen-scratch arena in
        ///              production, a test arena in tests); outlives the staging.
        explicit RuleIndexStaging(LbArena* arena)
            : wholeKeys(arena), originals(arena), copyOwners(arena),
              remArgsOwners(arena), lmv(arena), subkeys(arena), remArgs(arena) {}

        RuleIndexStaging(const RuleIndexStaging&) = delete;
        RuleIndexStaging& operator=(const RuleIndexStaging&) = delete;

        RuleIndexStagingSection wholeKeys;
        RuleIndexStagingSection originals;
        RuleIndexStagingSection copyOwners;
        RuleIndexStagingSection remArgsOwners;
        RuleIndexStagingSection lmv;
        RuleIndexStagingSection subkeys;
        RuleIndexStagingSection remArgs;

        /// @brief The hash-memory instance the staged records belong to
        ///        (`nullptr` while empty) — set by `RuleStagingRegistry::acquire`,
        ///        cleared by `clear`; the flush asserts it matches its map.
        const void* boundTo = nullptr;

        /// @brief Whether every section is empty (arena-free probe).
        /// @return `true` when nothing is staged.
        bool empty() const {
            return wholeKeys.empty() && originals.empty() && copyOwners.empty()
                && remArgsOwners.empty() && lmv.empty() && subkeys.empty()
                && remArgs.empty();
        }

        /// @brief Forget every section's records, return their pages, unbind.
        /// @return Nothing.
        void clear() {
            wholeKeys.clear();
            originals.clear();
            copyOwners.clear();
            remArgsOwners.clear();
            lmv.clear();
            subkeys.clear();
            remArgs.clear();
            boundTo = nullptr;
        }
    };

    /// @brief The per-worker-slot staging registry: a fixed pool of
    ///        `RuleIndexStaging` objects per gen-scratch slot, each bound to
    ///        the hash-memory instance it currently serves.
    ///
    /// @details
    /// Mirrors `ScratchArenaRegistry` (a `std::deque` for stable addresses,
    /// one program-start allocation). A window's installs `acquire` the
    /// staging bound to their target instance — or an empty one, which they
    /// bind — and the window's closer flushes every bound staging of the slot
    /// (`ExpressionAnalyzer::flushRuleIndexStaging`). An install may nest
    /// deposits into other LBs (an integration seed, a copy equality at an
    /// ancestor), each holding its own instance's staging until its exit, so
    /// the pool covers the nesting depth: `kPerSlot` bindings live at once,
    /// the overflow asserts (the Rule-19 tripwire). The slot is the caller's gen-scratch slot
    /// (`g_currentCoreId`, or the registry's last slot at a single-threaded
    /// seam), so the staging rides the same arena the install's other scratch
    /// does and dies with it.
    ///
    /// @invariant At every task end and every seam a reader of the rule
    ///            indexes reaches, every staging of the slot is empty
    ///            (`slotIsEmpty`) — a bound, non-empty staging outside its
    ///            window is a missed flush seam.
    /// @see `RuleIndexStaging`, `ScratchArenaRegistry`, `genScratchArenas`.
    class RuleStagingRegistry {
    public:
        /// @brief Stagings per slot — the bound on simultaneously bound
        ///        instances (nested installs across LBs times an LB's four
        ///        instances); overflow asserts.
        static constexpr int32_t kPerSlot = 32;

        /// @brief Create the stagings on the gen-scratch registry's arenas.
        /// @param gen The initialized gen-scratch registry.
        void init(ScratchArenaRegistry& gen) {
            assert(stagings_.empty() && "RuleStagingRegistry::init twice");
            assert(gen.initialized()
                && "RuleStagingRegistry::init before initGenScratchArenas");
            slots_ = gen.slotCount();
            for (unsigned s = 0; s < slots_; ++s)
                for (int32_t i = 0; i < kPerSlot; ++i)
                    stagings_.emplace_back(&gen.forSlot(s));
        }

        /// @brief Whether `init` ran.
        bool initialized() const { return !stagings_.empty(); }

        /// @brief The slot count (the gen-scratch registry's).
        unsigned slotCount() const { return slots_; }

        /// @brief The staging bound to @p hm on @p slot, or an empty one of the
        ///        slot bound to it now — the install's door.
        ///
        /// @param slot The caller's gen-scratch slot.
        /// @param hm   The target hash-memory instance.
        /// @return The staging serving @p hm; asserts when all four are bound
        ///         elsewhere (a window touching more instances than an LB has).
        RuleIndexStaging& acquire(unsigned slot, const void* hm) {
            assert(hm != nullptr);
            RuleIndexStaging* free = nullptr;
            for (int32_t i = 0; i < kPerSlot; ++i) {
                RuleIndexStaging& st = at(slot, i);
                if (st.boundTo == hm) return st;
                // Only an unbound staging is free: a bound one — even while
                // still empty — belongs to an open window's frame that will
                // stage into it; bindings change only at window closes.
                if (free == nullptr && st.boundTo == nullptr) free = &st;
            }
            assert(free != nullptr
                && "RuleStagingRegistry::acquire: every staging of the slot holds records for another instance - install nesting deeper than kPerSlot");
            free->boundTo = hm;
            return *free;
        }

        /// @brief The staging bound to @p hm on @p slot (possibly still
        ///        empty: an open frame that has not staged yet), or `nullptr`.
        ///
        /// @param slot The caller's gen-scratch slot.
        /// @param hm   The hash-memory instance.
        /// @return The bound staging, or `nullptr` when none is bound to it.
        RuleIndexStaging* boundTo(unsigned slot, const void* hm) {
            for (int32_t i = 0; i < kPerSlot; ++i) {
                RuleIndexStaging& st = at(slot, i);
                if (st.boundTo == hm) return &st;
            }
            return nullptr;
        }

        /// @brief The i-th staging of a slot (the flush's enumeration).
        /// @param slot The slot; `< slotCount()`.
        /// @param i    `0 <= i < kPerSlot`.
        /// @return The staging.
        RuleIndexStaging& at(unsigned slot, int32_t i) {
            assert(initialized() && slot < slots_ && i >= 0 && i < kPerSlot
                && "RuleStagingRegistry::at out of range");
            return stagings_[static_cast<std::size_t>(slot) * kPerSlot
                             + static_cast<std::size_t>(i)];
        }

        /// @brief Whether every staging of @p slot is empty — the seam probe.
        /// @param slot The slot.
        /// @return `true` when nothing is staged on the slot.
        bool slotIsEmpty(unsigned slot) {
            for (int32_t i = 0; i < kPerSlot; ++i)
                if (!at(slot, i).empty()) return false;
            return true;
        }

    private:
        std::deque<RuleIndexStaging> stagings_;   // deque: stable element addresses
        unsigned slots_ = 0;
    };

    /// @brief The process-wide rule-staging registry, created on first use
    ///        from the initialized gen-scratch registry (a thread-safe
    ///        magic static).
    /// @return The registry.
    inline RuleStagingRegistry& ruleStagings() {
        static RuleStagingRegistry& registry = []() -> RuleStagingRegistry& {
            static RuleStagingRegistry r;
            r.init(genScratchArenas());
            return r;
        }();
        return registry;
    }

}  // namespace gl
