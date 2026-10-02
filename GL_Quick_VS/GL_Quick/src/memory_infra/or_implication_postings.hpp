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

#include "typed_cold_map.hpp"
#include "paged_vector.hpp"

#include <cassert>
#include <cstdint>

namespace gl {

    /// @brief Pool-backed mutable postings for canonical OR matching.
    ///
    /// @details
    /// Each key owns a doubly linked list of distinct integer values. Exact
    /// edge lookup makes insertion and unlinking constant-time hash operations;
    /// retired nodes are immediately reused. Only a batched maintenance call
    /// compacts dead hash keys, so individual changes never shift a value run.
    /// Iteration order is internal: consumers sort their final work by decoded
    /// expression and validity before making observable deposits.
    ///
    /// The index is derived and owns no logical source state. Its private dirty
    /// flag does not dirty the owning block's canonical deload image. Canonical
    /// release clears it; reconstruction consumes the preserved source state.
    /// @tparam Key A fixed canonical key with a cold-map codec.
    /// @invariant Every live edge belongs to exactly one list and one pair key.
    /// @see OrImplications
    template<class Key>
    class OrPostingIndex {
        struct Pair { Key key; int64_t value; };
        struct Node { int64_t value; int32_t previous; int32_t next; };

        DirtyState dirty_ = DirtyState::Clean;
        ColdHashMap<PodKeyStore<Pair>, int32_t> pairs_;
        TypedColdMap<Key, int32_t> heads_;
        PagedVector<Node> nodes_;
        int32_t freeHead_ = -1;
        int32_t live_ = 0;
        int32_t deadPairs_ = 0;

    public:
        /// @brief Bind an empty derived index to one resident arena.
        /// @details The arena owns every payload page and outlives this index.
        /// @param arena The owning logic block or test arena.
        /// @return A constructed empty index.
        explicit OrPostingIndex(LbArena* arena)
            : pairs_(arena, &dirty_), heads_(arena, &dirty_),
              nodes_(arena, &dirty_) {}

        /// @brief Insert one distinct key/value edge.
        /// @details A duplicate is a defined no-op; a retired slot is reused.
        /// @param key The canonical expression, core, or token key.
        /// @param value The stable logical reference indexed under the key.
        /// @return True exactly when a new live edge was inserted.
        bool insert(Key key, int64_t value) {
            const Pair pair{key, value};
            typename decltype(pairs_)::InsertHint pairHint;
            const int32_t pairId = pairs_.lookupForInsert(pair, pairHint);
            if (pairId != 0 && pairs_.valueAt(pairId) >= 0) return false;

            int32_t slot;
            if (freeHead_ >= 0) {
                slot = freeHead_;
                freeHead_ = nodes_[slot].next;
            } else {
                slot = nodes_.size();
                nodes_.push_back(Node{0, -1, -1});
            }
            typename decltype(heads_)::InsertHint headHint;
            const int32_t headId = heads_.lookupForInsert(key, headHint);
            const int32_t head = headId == 0 ? -1 : heads_.valueAt(headId);
            nodes_.setAt(slot, Node{value, -1, head});
            if (head >= 0) {
                Node next = nodes_[head];
                next.previous = slot;
                nodes_.setAt(head, next);
            }
            if (headId != 0) heads_.setValueAt(headId, slot);
            else heads_.insertAt(key, slot, headHint);
            if (pairId != 0) {
                assert(deadPairs_ > 0);
                pairs_.setValueAt(pairId, slot);
                --deadPairs_;
            } else {
                pairs_.insertAt(pair, slot, pairHint);
            }
            ++live_;
            return true;
        }

        /// @brief Unlink one edge immediately from every lookup path.
        /// @details An absent edge is a defined no-op. The node enters the free
        /// list; batched maintenance later removes the dead pair-table key.
        /// @param key The posting key.
        /// @param value The exact logical reference to remove.
        /// @return True exactly when an existing live edge was removed.
        bool erase(Key key, int64_t value) {
            const int32_t pairId = pairs_.lookup(Pair{key, value});
            if (pairId == 0 || pairs_.valueAt(pairId) < 0) return false;
            const int32_t slot = pairs_.valueAt(pairId);
            const Node node = nodes_[slot];
            assert(node.value == value && live_ > 0);
            if (node.previous >= 0) {
                Node previous = nodes_[node.previous];
                previous.next = node.next;
                nodes_.setAt(node.previous, previous);
            } else {
                heads_.upsert(key, node.next);
            }
            if (node.next >= 0) {
                Node next = nodes_[node.next];
                next.previous = node.previous;
                nodes_.setAt(node.next, next);
            }
            pairs_.setValueAt(pairId, -1);
            nodes_.setAt(slot, Node{0, -1, freeHead_});
            freeHead_ = slot;
            --live_;
            ++deadPairs_;
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
            for (int32_t slot = *head; slot >= 0; slot = nodes_[slot].next)
                sink(nodes_[slot].value);
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
            for (int32_t slot = *head; slot >= 0; slot = nodes_[slot].next)
                if (!sink(nodes_[slot].value)) break;
        }

        /// @brief Compact retired lookup keys with amortized linear work.
        /// @details Maintenance is called once after a drain. A nonempty index
        /// compacts when at least half its pair keys are dead; the all-dead case
        /// releases every page. Current-key value probes during forward
        /// compaction refer to the unread slot, whose position has not moved.
        /// @return Void.
        void maintain() {
            if (live_ == 0) {
                clear();
                return;
            }
            if (deadPairs_ < 64 || deadPairs_ < live_) return;
            pairs_.eraseIf([&](const Pair& pair) {
                const int32_t id = pairs_.lookup(pair);
                assert(id > 0);
                return pairs_.valueAt(id) < 0;
            });
            heads_.eraseIf([&](const Key& key) {
                const int32_t* head = heads_.find(key);
                assert(head != nullptr);
                return *head < 0;
            });
            deadPairs_ = 0;
        }

        /// @brief Retire selected edges without changing surviving list order.
        ///
        /// @details Pair identifiers remain stable throughout unlinking. Batched
        /// maintenance runs after the final predicate call and reuses dead nodes.
        /// The predicate must not mutate this index.
        /// @tparam Remove Synchronous predicate accepting the key and value.
        /// @param remove True for an edge to retire.
        /// @return Nothing.
        template<class Remove>
        void eraseIf(Remove&& remove) {
            const int32_t count = pairs_.count();
            for (int32_t id = 1; id <= count; ++id) {
                if (pairs_.valueAt(id) < 0) continue;
                const Pair pair = pairs_.keyAt(id);
                if (remove(pair.key, pair.value)) {
                    const bool removed = erase(pair.key, pair.value);
                    assert(removed);
                }
            }
            maintain();
        }

        /// @brief Release all derived state before arena release or teardown.
        /// @details No node, hash key, or free-list reference survives this call.
        /// @return Void.
        void clear() {
            pairs_.resetToFresh();
            heads_.resetToFresh();
            nodes_.clear();
            freeHead_ = -1;
            live_ = 0;
            deadPairs_ = 0;
        }

        /// @brief Read the live edge count without walking the index.
        /// @details The count excludes retired pair keys and free nodes.
        /// @return Number of reachable key/value edges.
        int32_t size() const { return live_; }

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
            return pairs_.liveBytes() + heads_.liveBytes() + nodes_.liveBytes();
        }
    };

    using OrImplicationPostings = OrPostingIndex<int64_t>;
}
