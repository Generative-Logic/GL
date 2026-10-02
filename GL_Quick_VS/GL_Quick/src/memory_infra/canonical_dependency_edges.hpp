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

#include "or_implication_postings.hpp"

namespace gl {

/// @brief Store a canonical dependency once with both traversal directions.
///
/// @details Exact pair keys own stable recyclable nodes. Each node holds the
/// insertion revision and two newest-first linked lists; pair-table compaction
/// updates node-to-key identifiers without changing either traversal order.
class CanonicalDependencyEdges {
    struct Pair { int64_t token; int64_t record; };
    struct Node {
        int32_t pairId;
        int32_t revision;
        int32_t previous[2];
        int32_t next[2];
    };
    static_assert(sizeof(Pair) == 16 && sizeof(Node) == 24);
    DirtyState dirty_ = DirtyState::Clean;
    ColdHashMap<PodKeyStore<Pair>, int32_t> pairs_;
    TypedColdMap<int64_t, int32_t> memberHeads_, recordHeads_;
    PagedVector<Node> nodes_;
    int32_t freeHead_ = -1;
    int32_t live_ = 0;
    int32_t deadPairs_ = 0;

public:
    /// @brief Bind empty shared dependency storage to one arena.
    ///
    /// @details All payload and lookup pages remain on the owning logic block's
    /// arena. Construction allocates no logical edge or node.
    /// @param arena Resident owning arena, which outlives this store.
    /// @return An empty shared edge store.
    explicit CanonicalDependencyEdges(LbArena* arena)
        : pairs_(arena, &dirty_), memberHeads_(arena, &dirty_),
          recordHeads_(arena, &dirty_), nodes_(arena, &dirty_) {}

    /// @brief Insert a distinct dependency at both list heads.
    ///
    /// @details Duplicates leave the counter and every list unchanged. A retired
    /// pair reuses its key; a free node can serve a different pair. Fresh edges
    /// receive the next caller-owned logical revision before publication.
    /// @param token Exact packed dependency name and scope.
    /// @param record Exact packed owning record identity.
    /// @param revision Monotone logical revision, incremented only on insertion.
    /// @return True exactly when a new live edge is inserted.
    /// @invariant Revision overflow asserts before changing edge storage.
    bool insert(int64_t token, int64_t record, int32_t& revision) {
        const Pair pair{token, record};
        typename decltype(pairs_)::InsertHint hint;
        int32_t pairId = pairs_.lookupForInsert(pair, hint);
        if (pairId != 0 && pairs_.valueAt(pairId) >= 0) return false;
        assert(revision < INT32_MAX && "canonical dependency revision overflow");
        int32_t slot;
        if (freeHead_ >= 0) {
            slot = freeHead_;
            freeHead_ = nodes_[slot].next[0];
        } else {
            slot = nodes_.size();
            nodes_.push_back(Node{});
        }
        if (pairId != 0) {
            assert(deadPairs_ > 0);
            pairs_.setValueAt(pairId, slot);
            --deadPairs_;
        } else {
            pairs_.insertAt(pair, slot, hint);
            pairId = pairs_.count();
        }
        Node node{pairId, ++revision, {-1, -1}, {-1, -1}};
        const int64_t keys[2] = {token, record};
        for (int direction = 0; direction < 2; ++direction) {
            auto& heads = direction == 0 ? memberHeads_ : recordHeads_;
            const int32_t* head = heads.find(keys[direction]);
            node.next[direction] = head == nullptr ? -1 : *head;
            if (node.next[direction] >= 0) {
                Node next = nodes_[node.next[direction]];
                next.previous[direction] = slot;
                nodes_.setAt(node.next[direction], next);
            }
            heads.upsert(keys[direction], slot);
        }
        nodes_.setAt(slot, node);
        ++live_;
        return true;
    }

    /// @brief Remove an exact dependency from both traversal directions.
    ///
    /// @details Missing or retired pairs are defined no-ops. Removed slots enter
    /// the reusable free list; key compaction is deferred to maintain.
    /// @param token Packed dependency name and scope.
    /// @param record Packed owning record identity.
    /// @return True exactly when an existing live edge was removed.
    bool erase(int64_t token, int64_t record) {
        const int32_t pairId = pairs_.lookup(Pair{token, record});
        if (pairId == 0 || pairs_.valueAt(pairId) < 0) return false;
        const int32_t slot = pairs_.valueAt(pairId);
        const Node node = nodes_[slot];
        assert(node.pairId == pairId && node.revision > 0 && live_ > 0);
        const int64_t keys[2] = {token, record};
        for (int direction = 0; direction < 2; ++direction) {
            if (node.previous[direction] >= 0) {
                Node previous = nodes_[node.previous[direction]];
                previous.next[direction] = node.next[direction];
                nodes_.setAt(node.previous[direction], previous);
            } else {
                auto& heads = direction == 0 ? memberHeads_ : recordHeads_;
                heads.upsert(keys[direction], node.next[direction]);
            }
            if (node.next[direction] >= 0) {
                Node next = nodes_[node.next[direction]];
                next.previous[direction] = node.previous[direction];
                nodes_.setAt(node.next[direction], next);
            }
        }
        pairs_.setValueAt(pairId, -1);
        nodes_.setAt(slot, Node{0, 0, {-1, -1}, {freeHead_, -1}});
        freeHead_ = slot;
        --live_;
        ++deadPairs_;
        return true;
    }

    /// @brief Visit one ordered dependency prefix with its insertion revisions.
    ///
    /// @details Both directions preserve newest-first insertion order. Missing
    /// heads and empty lists yield no callbacks; false from the consumer ends
    /// the requested prefix. The consumer must not mutate this store.
    /// @tparam ByMember True selects token-to-record, false record-to-token.
    /// @tparam Sink Callback accepting the opposite key and insertion revision.
    /// @param key Exact head identity in the selected direction.
    /// @param sink Synchronous consumer returning whether to continue.
    /// @return Nothing.
    template<bool ByMember, class Sink>
    void forEachWhile(int64_t key, Sink&& sink) const {
        constexpr int direction = ByMember ? 0 : 1;
        const auto& heads = ByMember ? memberHeads_ : recordHeads_;
        const int32_t* head = heads.find(key);
        if (head == nullptr) return;
        for (int32_t slot = *head; slot >= 0; slot = nodes_[slot].next[direction]) {
            const Node node = nodes_[slot];
            assert(node.pairId > 0 && node.revision > 0);
            const Pair pair = pairs_.keyAt(node.pairId);
            assert((ByMember ? pair.token : pair.record) == key);
            if (!sink(ByMember ? pair.record : pair.token, node.revision)) break;
        }
    }

    /// @brief Reclaim retired pair keys without relocating live nodes.
    ///
    /// @details The existing posting-index maintenance threshold amortizes the
    /// stable key compaction. Every surviving node receives its new pair id;
    /// its revision and both linked-list orders remain unchanged. An all-dead
    /// store releases every page, including its recyclable nodes.
    /// @return Nothing.
    void maintain() {
        if (live_ == 0) { clear(); return; }
        if (deadPairs_ < 64 || deadPairs_ < live_) return;
        pairs_.eraseIf([&](const Pair& pair) {
            const int32_t id = pairs_.lookup(pair);
            assert(id > 0);
            return pairs_.valueAt(id) < 0;
        });
        for (int32_t id = 1; id <= pairs_.count(); ++id) {
            const int32_t slot = pairs_.valueAt(id);
            assert(slot >= 0);
            Node node = nodes_[slot];
            node.pairId = id;
            nodes_.setAt(slot, node);
        }
        const auto retireHeads = [](auto& heads) {
            heads.eraseIf([&](int64_t key) {
                const int32_t* head = heads.find(key);
                assert(head != nullptr);
                return *head < 0;
            });
        };
        retireHeads(memberHeads_);
        retireHeads(recordHeads_);
        deadPairs_ = 0;
    }

    /// @brief Remove selected edges and compact their retired lookup keys.
    ///
    /// @details Selection sees the exact stored token and record. Unlinking
    /// preserves surviving order; no key moves until selection is complete.
    /// @tparam Remove Pure predicate over token and record identities.
    /// @param remove True for an edge that should be retired.
    /// @return Nothing.
    template<class Remove>
    void eraseIf(Remove&& remove) {
        const int32_t count = pairs_.count();
        for (int32_t id = 1; id <= count; ++id) {
            if (pairs_.valueAt(id) < 0) continue;
            const Pair pair = pairs_.keyAt(id);
            if (remove(pair.token, pair.record)) {
                const bool removed = erase(pair.token, pair.record);
                assert(removed);
            }
        }
        maintain();
    }

    /// @brief Release every dependency and its derived storage.
    ///
    /// @details No free-node or list-head reference survives teardown. The
    /// caller owns the logical revision counter and decides when to reset it.
    /// @return Nothing.
    void clear() {
        pairs_.resetToFresh(); memberHeads_.resetToFresh(); recordHeads_.resetToFresh();
        nodes_.clear(); freeHead_ = -1; live_ = 0; deadPairs_ = 0;
    }

    /// @brief Count live edges, excluding retired pair keys and free nodes.
    ///
    /// @details This is a storage-accounting accessor with no proof-state effect.
    /// @return Current number of live exact dependencies.
    int32_t size() const { return live_; }

    /// @brief Account for the shared edge store's allocated arena pages.
    ///
    /// @details Includes key, index, head and node pages, including retained
    /// reusable capacity; it excludes the enclosing index's independent maps.
    /// @return Held storage bytes reported by the underlying cold containers.
    int64_t liveBytes() const {
        return pairs_.liveBytes() + memberHeads_.liveBytes()
            + recordHeads_.liveBytes() + nodes_.liveBytes();
    }
};
}
