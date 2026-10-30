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
#include "paged_hash_index.hpp"

namespace gl {
/// @brief Store exact posting endpoints and links once in packed columns.
/// @details Buckets contain stable slot identifiers. Backward-shift deletion
/// changes only derived buckets; live edge slots and list order never move.
/// @tparam Key Fixed key with a unique object representation.
/// @tparam LinkWords Number of signed link and revision words per edge.
/// @invariant Word zero is -2 only for a retired slot; word one then links free slots.
template<class Key, int LinkWords>
class FusedPostingStorage {
    struct Pair { Key key; int64_t value; };
    static_assert(std::has_unique_object_representations_v<Pair>);
    static_assert(sizeof(Pair) == sizeof(Key) + sizeof(int64_t));
    static_assert(LinkWords >= 2);
    DirtyState dirty_ = DirtyState::Clean;
    PagedVector<Key> keys_;
    PagedVector<int64_t> values_;
    PagedVector<int32_t> links_;
    PagedHashIndex buckets_;
    int32_t freeHead_ = -1;
    int32_t live_ = 0;

    /// @brief Hash an exact endpoint pair.
    /// @details The raw-byte digest matches the fixed-key cold-map digest.
    /// @param key Posting key.
    /// @param value Opposite endpoint.
    /// @return Full 64-bit hash.
    static uint64_t hash(Key key, int64_t value) {
        const Pair pair{key, value};
        return hashSpan(StrSpan(reinterpret_cast<const char*>(&pair), sizeof(pair)));
    }

    /// @brief Place an already initialized live slot in the derived buckets.
    /// @details Probes the first empty slot without relocating edge columns.
    /// @param slot Live stable slot.
    /// @return Nothing.
    void place(int32_t slot) {
        const uint64_t mask = buckets_.capacity() - 1;
        uint64_t at = hash(keys_[slot], values_[slot]) & mask;
        while (buckets_.at(static_cast<int32_t>(at)) != 0) at = (at + 1) & mask;
        buckets_.set(static_cast<int32_t>(at), slot + 1);
    }

    /// @brief Rebuild lookup buckets for all live stable slots.
    /// @details Capacity follows the reference half-full policy; retired slots
    /// are excluded from lookup while retaining their reusable column storage.
    /// @return Nothing.
    void rebuild() {
        int32_t capacity = 64;
        while (static_cast<int64_t>(capacity) < static_cast<int64_t>(live_) * 2 + 2) {
            assert(capacity <= INT32_MAX / 2);
            capacity <<= 1;
        }
        buckets_.reset(capacity);
        for (int32_t slot = 0; slot < slots(); ++slot) if (isLive(slot)) place(slot);
    }

public:
    /// @brief Bind derived storage to its owning arena.
    /// @details Construction allocates no pages. The arena outlives every column.
    /// @param arena Owning resident arena.
    /// @return Empty constructed storage.
    explicit FusedPostingStorage(LbArena* arena)
        : keys_(arena, &dirty_), values_(arena, &dirty_), links_(arena, &dirty_), buckets_(arena) {}

    /// @brief Read the retained stable-slot high-water.
    /// @details Includes free slots available for immediate reuse.
    /// @return Retained slot count.
    int32_t slots() const { return keys_.size(); }

    /// @brief Read the exact live edge count.
    /// @details Free slots contribute no live edge.
    /// @return Live edge count.
    int32_t size() const { return live_; }

    /// @brief Test a retained slot's live marker.
    /// @details The first previous-link word doubles as the free-slot marker.
    /// @param slot Retained slot index.
    /// @return True for a live edge.
    bool isLive(int32_t slot) const { return word(slot, 0) != -2; }

    /// @brief Read one endpoint key by stable slot.
    /// @details Returns a fixed-size value without a borrowed page reference.
    /// @param slot Live slot.
    /// @return Stored key.
    Key key(int32_t slot) const { assert(isLive(slot)); return keys_[slot]; }

    /// @brief Read the opposite endpoint by stable slot.
    /// @details Returns the complete unchanged 64-bit identity.
    /// @param slot Live slot.
    /// @return Opposite endpoint.
    int64_t value(int32_t slot) const { assert(isLive(slot)); return values_[slot]; }

    /// @brief Read one packed link or revision word.
    /// @details Column addressing packs all words without per-record page padding.
    /// @param slot Retained slot index.
    /// @param index Word index within an edge.
    /// @return Stored signed word.
    int32_t word(int32_t slot, int index) const {
        assert(slot >= 0 && slot < slots() && index >= 0 && index < LinkWords);
        return links_[slot * LinkWords + index];
    }

    /// @brief Update one packed link or revision word.
    /// @details Neighbor edits preserve every other edge field.
    /// @param slot Retained slot index.
    /// @param index Word index within an edge.
    /// @param value New signed word.
    /// @return Nothing.
    void setWord(int32_t slot, int index, int32_t value) {
        assert(slot >= 0 && slot < slots() && index >= 0 && index < LinkWords);
        links_.setAt(slot * LinkWords + index, value);
    }

    /// @brief Find an exact live endpoint pair.
    /// @details A missing pair is a defined result; collisions compare all bytes.
    /// @param key Posting key.
    /// @param value Opposite endpoint.
    /// @return Stable slot, or -1 for a defined miss.
    int32_t find(Key key, int64_t value) const {
        if (buckets_.empty()) return -1;
        const uint64_t mask = buckets_.capacity() - 1;
        uint64_t at = hash(key, value) & mask;
        while (true) {
            const int32_t entry = buckets_.at(static_cast<int32_t>(at));
            if (entry == 0) return -1;
            const int32_t slot = entry - 1;
            const Key stored = keys_[slot];
            if (values_[slot] == value && std::memcmp(&stored, &key, sizeof(Key)) == 0) return slot;
            at = (at + 1) & mask;
        }
    }

    /// @brief Allocate and index a caller-proven absent endpoint pair.
    /// @details Reuses a retired slot or appends fully packed columns. All link
    /// words start at -1 and the caller publishes its list links synchronously.
    /// @param key Absent posting key.
    /// @param value Opposite endpoint.
    /// @return New live stable slot.
    int32_t insertAbsent(Key key, int64_t value) {
        assert(find(key, value) < 0);
        int32_t slot;
        if (freeHead_ >= 0) {
            slot = freeHead_; freeHead_ = word(slot, 1);
            keys_.setAt(slot, key); values_.setAt(slot, value);
            for (int i = 0; i < LinkWords; ++i) setWord(slot, i, -1);
        } else {
            slot = slots(); assert(slot < INT32_MAX / LinkWords);
            keys_.push_back(key); values_.push_back(value);
            for (int i = 0; i < LinkWords; ++i) links_.push_back(-1);
        }
        ++live_;
        if (buckets_.empty() || static_cast<int64_t>(live_) * 2 >= buckets_.capacity()) rebuild();
        else place(slot);
        assert(find(key, value) == slot);
        return slot;
    }

    /// @brief Retire an already unlinked edge without moving surviving slots.
    /// @details Backward shifts only lookup buckets whose probe path crosses the
    /// hole. The retired edge enters the reusable free list immediately.
    /// @param slot Live slot whose list neighbors have already been updated.
    /// @return Nothing.
    void retire(int32_t slot) {
        assert(isLive(slot) && live_ > 0);
        const uint64_t mask = buckets_.capacity() - 1;
        uint64_t hole = hash(keys_[slot], values_[slot]) & mask;
        while (buckets_.at(static_cast<int32_t>(hole)) != slot + 1) {
            assert(buckets_.at(static_cast<int32_t>(hole)) != 0);
            hole = (hole + 1) & mask;
        }
        uint64_t at = (hole + 1) & mask;
        while (buckets_.at(static_cast<int32_t>(at)) != 0) {
            const int32_t entry = buckets_.at(static_cast<int32_t>(at));
            const int32_t other = entry - 1;
            const uint64_t home = hash(keys_[other], values_[other]) & mask;
            if (((at - home) & mask) >= ((at - hole) & mask)) {
                buckets_.set(static_cast<int32_t>(hole), entry); hole = at;
            }
            at = (at + 1) & mask;
        }
        buckets_.set(static_cast<int32_t>(hole), 0);
        setWord(slot, 0, -2); setWord(slot, 1, freeHead_);
        freeHead_ = slot; --live_;
    }

    /// @brief Release every derived column and bucket page.
    /// @details Called before canonical arena release; raw images retain all columns.
    /// @return Nothing.
    void clear() {
        keys_.clear(); values_.clear(); links_.clear(); buckets_.clear();
        freeHead_ = -1; live_ = 0;
    }

    /// @brief Account for all retained payload and lookup pages.
    /// @details Named census children are disjoint and include free-slot capacity.
    /// @return Retained bytes including page directories.
    int64_t liveBytes() const {
        return peak_snapshot::measure("keys_", keys_, [&] { return keys_.liveBytes(); })
            + peak_snapshot::measure("values_", values_, [&] { return values_.liveBytes(); })
            + peak_snapshot::measure("links_", links_, [&] { return links_.liveBytes(); })
            + peak_snapshot::measure("buckets_", buckets_, [&] { return buckets_.liveBytes(); });
    }
};
}
