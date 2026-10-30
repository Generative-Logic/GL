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

#include "fused_posting_storage.hpp"
#include "paged_vector.hpp"

#include <cassert>
#include <cstdint>

namespace gl {

    /// @brief Pool-backed mutable postings for canonical OR matching.
    ///
    /// @details
    /// Each key owns a doubly linked list of distinct integer values. Exact
    /// edge lookup makes insertion and unlinking constant-time hash operations;
    /// retired slots are immediately reused. Lookup buckets point directly at the
    /// packed endpoint/link columns; deletion never shifts a live edge slot.
    /// Iteration order is internal: consumers sort their final work by decoded
    /// expression and validity before making observable deposits.
    ///
    /// The index is derived and owns no logical source state. Its private dirty
    /// flag does not dirty the owning block's canonical deload image. Canonical
    /// release clears it; reconstruction consumes the preserved source state.
    /// @tparam Key A fixed canonical key with a cold-map codec.
    /// @invariant Every live edge belongs to exactly one list and one lookup bucket.
    /// @see OrImplications
    template<class Key>
    class OrPostingIndex {
        DirtyState dirty_ = DirtyState::Clean;
        TypedColdMap<Key, int32_t> heads_;
        FusedPostingStorage<Key, 2> edges_;
        int32_t retiredSinceMaintenance_ = 0;

    public:
        /// @brief Bind an empty derived index to one resident arena.
        /// @details The arena owns every payload page and outlives this index.
        /// @param arena The owning logic block or test arena.
        /// @return A constructed empty index.
        explicit OrPostingIndex(LbArena* arena)
            : heads_(arena, &dirty_), edges_(arena) {}

        /// @brief Insert one distinct key/value edge.
        /// @details A duplicate is a defined no-op; a retired slot is reused.
        /// @param key The canonical expression, core, or token key.
        /// @param value The stable logical reference indexed under the key.
        /// @return True exactly when a new live edge was inserted.
        bool insert(Key key, int64_t value) {
        if (edges_.find(key, value) >= 0) return false;
        const int32_t slot = edges_.insertAbsent(key, value);
        typename decltype(heads_)::InsertHint hint;
        const int32_t id = heads_.lookupForInsert(key, hint);
        const int32_t head = id == 0 ? -1 : heads_.valueAt(id);
        edges_.setWord(slot, 1, head);
        if (head >= 0) edges_.setWord(head, 0, slot);
        if (id != 0) heads_.setValueAt(id, slot);
        else heads_.insertAt(key, slot, hint);
        return true;
    }

        /// @brief Unlink one edge immediately from every lookup path.
        /// @details An absent edge is a defined no-op. The node enters the free
        /// list; backward-shift deletion immediately removes its lookup entry.
        /// @param key The posting key.
        /// @param value The exact logical reference to remove.
        /// @return True exactly when an existing live edge was removed.
        bool erase(Key key, int64_t value) {
        const int32_t slot = edges_.find(key, value);
        if (slot < 0) return false;
        const int32_t previous = edges_.word(slot, 0), next = edges_.word(slot, 1);
        if (previous >= 0) edges_.setWord(previous, 1, next);
        else heads_.upsert(key, next);
        if (next >= 0) edges_.setWord(next, 0, previous);
        edges_.retire(slot);
        assert(retiredSinceMaintenance_ < INT32_MAX);
        ++retiredSinceMaintenance_;
        return true;
    }

        /// @brief Visit exactly the live values under one key.
        /// @details The callback must not mutate this index. Missing keys and
        /// empty lists produce no callbacks; no retired node is visited.
        /// @tparam Sink A synchronous callback accepting an int64_t reference.
        /// @param key The exact index key.
        /// @param sink The consumer of each live value.
        /// @return Void.
        template<class Sink>
        void forEach(Key key, Sink&& sink) const {
        const int32_t* head = heads_.find(key);
        if (head == nullptr) return;
        for (int32_t slot = *head; slot >= 0; slot = edges_.word(slot, 1))
            sink(edges_.value(slot));
    }

        /// @brief Visit a live posting prefix until its consumer stops.
        /// @details New edges are prepended; erasure preserves the order of
        /// surviving edges. The callback must not mutate this index. An absent
        /// key or empty list has no visits.
        /// @tparam Sink Synchronous callback returning whether to continue.
        /// @param key Exact posting key.
        /// @param sink Consumer of each live value, newest insertion first.
        /// @return Nothing.
        template<class Sink>
        void forEachWhile(Key key, Sink&& sink) const {
        const int32_t* head = heads_.find(key);
        if (head == nullptr) return;
        for (int32_t slot = *head; slot >= 0; slot = edges_.word(slot, 1))
            if (!sink(edges_.value(slot))) break;
    }

        /// @brief Remove empty list heads with amortized linear work.
        /// @details Maintenance is called once after a drain. A nonempty index
        /// cleans after at least 64 retirements and at least as many retirements as
        /// live edges; the all-dead case releases every page. Edge slots and
        /// newest-first list order remain unchanged.
        /// @return Void.
        void maintain() {
        if (edges_.size() == 0) { clear(); return; }
        if (retiredSinceMaintenance_ < 64 || retiredSinceMaintenance_ < edges_.size()) return;
        heads_.eraseIf([&](const Key& key) {
            const int32_t* head = heads_.find(key);
            assert(head != nullptr);
            return *head < 0;
        });
        retiredSinceMaintenance_ = 0;
    }

        /// @brief Retire selected edges without changing surviving list order.
        ///
        /// @details Stable edge slots remain fixed throughout unlinking. Batched
        /// head maintenance runs after the final predicate call.
        /// The predicate must not mutate this index.
        /// @tparam Remove Synchronous predicate accepting the key and value.
        /// @param remove True for an edge to retire.
        /// @return Nothing.
        template<class Remove>
        void eraseIf(Remove&& remove) {
        const int32_t count = edges_.slots();
        for (int32_t slot = 0; slot < count; ++slot) {
            if (!edges_.isLive(slot)) continue;
            const Key key = edges_.key(slot);
            const int64_t value = edges_.value(slot);
            if (remove(key, value)) { const bool removed = erase(key, value); assert(removed); }
        }
        maintain();
    }

        /// @brief Release all derived state before arena release or teardown.
        /// @details No edge, lookup bucket, or free-list reference survives this call.
        /// @return Void.
        void clear() {
        heads_.resetToFresh(); edges_.clear(); retiredSinceMaintenance_ = 0;
    }

        /// @brief Read the live edge count without walking the index.
        /// @details The count excludes free edge slots.
        /// @return Number of reachable key/value edges.
        int32_t size() const { return edges_.size(); }

        /// @brief Test whether an exact key has any live posting.
        /// @details Empty and never-seen keys are the same defined miss.
        /// @param key The canonical posting key.
        /// @return True exactly when a live value is reachable from the key.
        bool containsKey(const Key& key) const {
            const int32_t* head = heads_.find(key);
            return head != nullptr && *head >= 0;
        }

        /// @brief Read retained posting storage for memory accounting.
        /// @details Includes live and not-yet-compacted storage in all columns.
        /// @return Bytes reported by the underlying pool-backed containers.
        int64_t liveBytes() const {
        return peak_snapshot::measure("heads_", heads_, [&] { return heads_.liveBytes(); })
            + peak_snapshot::measure("edges_", edges_, [&] { return edges_.liveBytes(); });
    }
    };

    using OrImplicationPostings = OrPostingIndex<int64_t>;
}
