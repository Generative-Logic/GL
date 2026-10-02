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

#include "int_encoded_expr.hpp"
#include "paged_vector.hpp"
#include "paged_hash_index.hpp"
#include <cstring>
#include <limits>

namespace gl {
    /// @brief Own immutable expression records shared by one block's ordered lists.
    ///
    /// @details Exact byte identity includes every argument and scope field. Slots
    /// remain stable while referenced; zero-reference slots are recycled. The
    /// canonical stream is supplied by the lists, while raw eviction retains
    /// this arena-backed store and its reference counts together.
    /// @invariant Only live list references keep a record logically present.
    class SharedExpressionStore {
        struct Slot {
            IntEncodedExpr value;
            int32_t references;
            int32_t nextFree;
            uint32_t hash;
        };
        LbArena* arena_;
        DirtyState storageDirty_ = DirtyState::Clean;
        PagedVector<Slot> slots_;
        PagedHashIndex index_;
        int32_t freeHead_ = -1;
        int32_t live_ = 0;
        int32_t occupied_ = 0;

        /// @brief Hash all observable record bytes.
        ///
        /// @details IntEncodedExpr has no padding; complete equality is checked
        /// after a hash match, so collisions never merge unequal expressions.
        /// @param value Record to hash.
        /// @return Stable unsigned content hash.
        static uint32_t hash(const IntEncodedExpr& value) {
            const auto* bytes = reinterpret_cast<const unsigned char*>(&value);
            uint32_t result = 2166136261u;
            for (size_t i = 0; i < sizeof(value); ++i)
                result = (result ^ bytes[i]) * 16777619u;
            return result;
        }

        /// @brief Rebuild the derived lookup table over stable live slots.
        ///
        /// @details Clears tombstones without changing any record identifier.
        /// @param capacity Positive power-of-two bucket count.
        /// @return Nothing.
        void rebuild(int32_t capacity) {
            assert(capacity > live_ && (capacity & (capacity - 1)) == 0);
            index_.reset(capacity);
            for (int32_t id = 0; id < slots_.size(); ++id) {
                const Slot& row = slots_[id];
                if (row.references == 0) continue;
                int32_t bucket = static_cast<int32_t>(row.hash & (capacity - 1));
                while (index_.at(bucket) != 0) bucket = (bucket + 1) & (capacity - 1);
                index_.set(bucket, id + 1);
            }
            occupied_ = live_;
        }
    public:
        /// @brief Bind a lazy record store to its block arena.
        ///
        /// @details Construction consumes no pages. Lists must be destroyed first.
        /// @param arena Arena that owns every record and lookup page.
        /// @return Nothing.
        explicit SharedExpressionStore(LbArena* arena)
            : arena_(arena), slots_(arena, &storageDirty_), index_(arena) {
            assert(arena != nullptr);
        }
        SharedExpressionStore(const SharedExpressionStore&) = delete;
        SharedExpressionStore& operator=(const SharedExpressionStore&) = delete;

        /// @brief Retain one exact record and return its stable slot identifier.
        ///
        /// @details A hit increments its reference count. A miss fills a recycled
        /// slot or appends one, growing or cleaning the hash table when required.
        /// @param value Full record, possibly borrowed from this store.
        /// @return Nonnegative identifier owned by one new list reference.
        int32_t retain(const IntEncodedExpr& value) {
            const IntEncodedExpr copy = value;
            const uint32_t code = hash(copy);
            int32_t capacity = index_.capacity();
            if (capacity == 0) { rebuild(64); capacity = 64; }
            if (static_cast<int64_t>(occupied_ + 1) * 2 >= capacity) {
                const bool grow = static_cast<int64_t>(live_ + 1) * 2 >= capacity;
                assert(!grow || capacity <= std::numeric_limits<int32_t>::max() / 2);
                rebuild(grow ? capacity * 2 : capacity);
                capacity = index_.capacity();
            }
            int32_t bucket = static_cast<int32_t>(code & (capacity - 1));
            int32_t vacant = -1;
            while (index_.at(bucket) != 0) {
                const int32_t entry = index_.at(bucket);
                if (entry < 0) {
                    if (vacant < 0) vacant = bucket;
                } else {
                    Slot row = slots_[entry - 1];
                    if (row.hash == code && std::memcmp(&row.value, &copy, sizeof(copy)) == 0) {
                        assert(row.references > 0 && row.references < std::numeric_limits<int32_t>::max());
                        ++row.references;
                        slots_.setAt(entry - 1, row);
                        return entry - 1;
                    }
                }
                bucket = (bucket + 1) & (capacity - 1);
            }
            if (vacant >= 0) bucket = vacant;
            else ++occupied_;
            const Slot row{copy, 1, -1, code};
            int32_t id;
            if (freeHead_ >= 0) {
                id = freeHead_;
                assert(slots_[id].references == 0);
                freeHead_ = slots_[id].nextFree;
                slots_.setAt(id, row);
            } else {
                id = slots_.size();
                assert(id < std::numeric_limits<int32_t>::max() - 1);
                slots_.push_back(row);
            }
            ++live_;
            index_.set(bucket, id + 1);
            return id;
        }

        /// @brief Drop one list reference and recycle its last-owned record.
        ///
        /// @details A tombstone preserves probe chains. Once all lists are empty,
        /// all store pages are returned immediately. Partial holes are reusable.
        /// @param id Live identifier owned by the releasing list.
        /// @return Nothing.
        void drop(int32_t id) {
            assert(id >= 0 && id < slots_.size());
            Slot row = slots_[id];
            assert(row.references > 0);
            --row.references;
            if (row.references == 0) {
                const int32_t mask = index_.capacity() - 1;
                int32_t bucket = static_cast<int32_t>(row.hash & mask);
                while (index_.at(bucket) != id + 1) {
                    assert(index_.at(bucket) != 0);
                    bucket = (bucket + 1) & mask;
                }
                index_.set(bucket, -1);
                row.nextFree = freeHead_;
                freeHead_ = id;
                --live_;
            }
            slots_.setAt(id, row);
            if (live_ == 0) {
                slots_.clear();
                index_.clear();
                freeHead_ = -1;
                occupied_ = 0;
            }
        }

        /// @brief Read a live immutable record through its stable identifier.
        ///
        /// @details The reference survives growth but not its last release or arena relocation.
        /// @param id Live record identifier.
        /// @return Borrowed record on the resident arena.
        const IntEncodedExpr& at(int32_t id) const {
            assert(id >= 0 && id < slots_.size() && slots_[id].references > 0);
            return slots_[id].value;
        }
        /// @brief Count distinct records currently owned by lists.
        ///
        /// @details Recycled slots are excluded from logical accounting.
        /// @return Live distinct record count.
        int32_t count() const { assert(arena_->resident()); return live_; }
        /// @brief Account physical record and lookup pages exactly once.
        ///
        /// @details Includes reusable holes and directory pages retained by the store.
        /// @return Allocated bytes on the owning arena.
        int64_t liveBytes() const { return slots_.liveBytes() + index_.liveBytes(); }
        /// @brief Account retained record slots without their page rounding.
        ///
        /// @details Includes metadata and reusable slots; list references are separate.
        /// @return Resident record-content bytes.
        int64_t residentContentBytes() const { return static_cast<int64_t>(slots_.size()) * sizeof(Slot); }
        /// @brief Account the store's derived lookup pages.
        ///
        /// @details This index is shared by all four lists and counted only once.
        /// @return Allocated lookup bytes.
        int64_t indexBytes() const { return index_.liveBytes(); }

    };

    /// @brief Preserve a statement list's order using shared record references.
    ///
    /// @details Each list owns one reference per position, including duplicates.
    /// Mutation preserves the former vector contract. Canonical serialization
    /// expands full records in list order, retaining the existing deload format.
    /// @invariant Sharing never changes scope, order, membership or record bytes.
    class SharedExpressionList {
        LbArena* arena_;
        SharedExpressionStore* store_;
        PagedVector<int32_t> ids_;
    public:
        using value_type = IntEncodedExpr;
        /// @brief Construct an empty list bound to its block's shared store.
        ///
        /// @details No pages are allocated until the first append.
        /// @param arena Resident arena shared with the store.
        /// @param dirty Logical block content-change flag.
        /// @param store Store that outlives this list.
        /// @return Nothing.
        SharedExpressionList(LbArena* arena, DirtyState* dirty, SharedExpressionStore* store)
            : arena_(arena), store_(store), ids_(arena, dirty) { assert(store != nullptr); }
        SharedExpressionList(const SharedExpressionList&) = delete;
        /// @brief Release resident references before destroying the list.
        ///
        /// @details A raw-deloaded arena has no accessible pages; its member
        /// containers discard only bookkeeping through their existing destructors.
        /// @return Nothing.
        ~SharedExpressionList() { if (arena_->resident()) clear(); }
        /// @brief Read the logical number of positions.
        ///
        /// @details Residency is checked by the reference vector.
        /// @return Number of records, including repeated positions.
        int32_t size() const { return ids_.size(); }
        /// @brief Test whether this list owns no positions.
        ///
        /// @details Other lists may still own shared records.
        /// @return True exactly when size is zero.
        bool empty() const { return ids_.empty(); }
        /// @brief Resolve a position to its immutable record.
        ///
        /// @details The list position and shared identifier are private storage details.
        /// @param index Existing logical position.
        /// @return Borrowed full record.
        const IntEncodedExpr& operator[](int32_t index) const { return store_->at(ids_[index]); }
        /// @brief Append one independently owned position.
        ///
        /// @details Equal bytes share storage while their positions remain distinct.
        /// @param value Record to append, including its complete scope metadata.
        /// @return Nothing.
        void push_back(const IntEncodedExpr& value) { ids_.push_back(store_->retain(value)); }
        /// @brief Replace one position without invalidating shared peers.
        ///
        /// @details Retaining first also supports a value borrowed from the old record.
        /// @param index Existing position.
        /// @param value Replacement record.
        /// @return Nothing.
        void setAt(int32_t index, const IntEncodedExpr& value) {
            const int32_t old = ids_[index];
            const int32_t id = store_->retain(value);
            ids_.setAt(index, id);
            store_->drop(old);
        }
        /// @brief Remove one position and preserve survivor order.
        ///
        /// @details Only the last reference recycles a shared record.
        /// @param index Position to remove.
        /// @return Nothing.
        void erase(int32_t index) { const int32_t id = ids_[index]; ids_.erase(index); store_->drop(id); }
        /// @brief Retain only a prefix of the logical list.
        ///
        /// @details Removed positions release their own references exactly once.
        /// @param count Retained prefix length in the existing size range.
        /// @return Nothing.
        void truncate(int32_t count) {
            assert(count >= 0 && count <= size());
            for (int32_t i = count; i < size(); ++i) store_->drop(ids_[i]);
            ids_.truncate(count);
        }
        /// @brief Filter positions while preserving the original survivor order.
        ///
        /// @details The predicate observes full immutable records once per position.
        /// @param predicate True for positions to remove.
        /// @param onErased Callback receiving each removed original position.
        /// @return Number of removed positions.
        template<class Predicate, class OnErased> int32_t eraseIf(Predicate predicate, OnErased onErased) {
            const int32_t oldSize = size();
            int32_t write = 0;
            for (int32_t read = 0; read < oldSize; ++read) {
                const int32_t id = ids_[read];
                if (predicate(store_->at(id))) { onErased(read); store_->drop(id); }
                else { if (write != read) ids_.setAt(write, id); ++write; }
            }
            ids_.truncate(write);
            return oldSize - write;
        }
        /// @brief Filter positions when no erased-position callback is needed.
        ///
        /// @details Delegates to the ordered callback form with an empty observer.
        /// @param predicate True for positions to remove.
        /// @return Number of removed positions.
        template<class Predicate> int32_t eraseIf(Predicate predicate) {
            return eraseIf(predicate, [](int32_t) {});
        }
        /// @brief Empty this list and release all its reference pages.
        ///
        /// @details Peer lists retain their own records independently.
        /// @return Nothing.
        void clear() { truncate(0); ids_.clear(); }
        /// @brief Release this list at canonical deload or teardown.
        ///
        /// @details Expanding the canonical stream precedes this operation.
        /// @return Nothing.
        void release() { clear(); }
        /// @brief Deep-copy another list without importing foreign identifiers.
        ///
        /// @details Copies full records in order into the destination's store.
        /// @param other Source list, possibly in a different arena.
        /// @return This list.
        SharedExpressionList& operator=(const SharedExpressionList& other) {
            if (this == &other) return *this;
            clear();
            for (int32_t i = 0; i < other.size(); ++i) push_back(other[i]);
            return *this;
        }
        /// @brief Replace this list from a separate ordered record container.
        ///
        /// @details Supports existing scratch vectors without a heap intermediate.
        /// @param other Independently owned source records.
        /// @return This list.
        template<class Rows> SharedExpressionList& operator=(const Rows& other) {
            clear();
            for (size_t i = 0; i < static_cast<size_t>(other.size()); ++i) push_back(other[static_cast<int32_t>(i)]);
            return *this;
        }
        /// @brief Copy a logical run into a caller-owned contiguous buffer.
        ///
        /// @details List order is preserved across arbitrary shared identifiers.
        /// @param at First position.
        /// @param count Number of positions.
        /// @param destination Buffer with room for every copied record.
        /// @return Nothing.
        void copyRunTo(int32_t at, int32_t count, IntEncodedExpr* destination) const {
            assert(at >= 0 && count >= 0 && static_cast<int64_t>(at) + count <= size());
            for (int32_t i = 0; i < count; ++i) destination[i] = (*this)[at + i];
        }
        /// @brief Append full canonical row bytes to an external serializer buffer.
        ///
        /// @details This boundary preserves the former vector's element stream.
        /// @param destination Serializer-owned byte buffer.
        /// @param from First logical position to serialize.
        /// @return Nothing.
        void appendSpanBytes(std::vector<char>& destination, int64_t from) const {
            assert(from >= 0 && from <= size());
            for (int32_t i = static_cast<int32_t>(from); i < size(); ++i) {
                const char* bytes = reinterpret_cast<const char*>(&(*this)[i]);
                destination.insert(destination.end(), bytes, bytes + sizeof(IntEncodedExpr));
            }
        }
        /// @brief Restore canonical full records into this list's shared store.
        ///
        /// @details Unaligned input is copied through a local record before interning.
        /// @param bytes Contiguous serialized full records.
        /// @param count Number of records to append.
        /// @return Nothing.
        void bulkAppendBytes(const char* bytes, int64_t count) {
            assert(count >= 0 && count <= std::numeric_limits<int32_t>::max() - size());
            for (int64_t i = 0; i < count; ++i) {
                IntEncodedExpr value;
                std::memcpy(&value, bytes + i * sizeof(value), sizeof(value));
                push_back(value);
            }
        }
        /// @brief Account this list's reference pages without counting shared records.
        ///
        /// @details The owner reports the store separately, once per logic block.
        /// @return Physical reference storage bytes.
        int64_t liveBytes() const { return ids_.liveBytes(); }
        /// @brief Account reference content instead of expanded canonical records.
        ///
        /// @details Canonical deload keeps the full-record value_type; resident
        /// measurement must count four bytes per list position instead.
        /// @return Reference content bytes, excluding page rounding.
        int64_t residentContentBytes() const { return static_cast<int64_t>(size()) * sizeof(int32_t); }

        /// @brief Count physical reference pages for container diagnostics.
        ///
        /// @details Shared record pages belong to the separate store owner.
        /// @return Reference data-page count.
        int64_t pagesHeld() const { return ids_.pagesHeld(); }
    };
}
