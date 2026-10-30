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

/// @brief Register canonical records and store each record's name dependencies once.
///
/// @details The record map holds every registered record with its newest
/// dependency slot, or with the token-free or retired state. The member map
/// holds every inserted dependency token with its newest slot. A slot stores
/// both endpoints as identifiers in those maps, its insertion revision, a
/// doubly linked member list and a singly linked record list, all newest
/// first, in 24 bytes across two packed columns. An exact pair is found on its
/// record's short list, so no pair hash index exists. Key compaction rewrites
/// endpoint identifiers by walking the surviving lists; live slots, links and
/// revisions never move. See D-487.
class CanonicalDependencyEdges {
    /// @brief The member-traversal fields of one slot.
    ///
    /// @details One 16-byte element serves a member walk with a single page
    /// resolution. A retired slot has revision zero and links the free list
    /// through memberNext.
    struct ForwardWords { int32_t memberNext, revision, member, record; };

    /// @brief The remaining link fields of one slot.
    ///
    /// @details The member list is doubly linked for constant-time unlinking
    /// from long token lists; the record list is singly linked because a
    /// record's list is short and is walked from its head.
    struct BackwardWords { int32_t memberPrevious, recordNext; };

    using HeadMap = TypedColdMap<int64_t, int32_t>;

    static constexpr int32_t kNoEdge = -1;
    static constexpr int32_t kRetired = -2;

    HeadMap members_, records_;
    PagedVector<ForwardWords> forward_;
    PagedVector<BackwardWords> backward_;
    int32_t freeHead_ = -1;
    int32_t live_ = 0;

    /// @brief Find an exact pair on its record's newest-first list.
    ///
    /// @details The walk also reports the list predecessor that unlinking a
    /// singly linked record list requires. A token-free or retired record has
    /// no list. A missing pair is a defined result.
    /// @param memberId Member-map identifier of the dependency token.
    /// @param recordId Record-map identifier of the owning record.
    /// @param previous [out] Record-list predecessor of the found slot, or -1
    ///                 when the slot is the record's head.
    /// @return Live slot of the pair, or -1 when the pair is absent.
    int32_t findOnRecord(int32_t memberId, int32_t recordId, int32_t& previous) const {
        previous = -1;
        for (int32_t slot = records_.valueAt(recordId); slot >= 0; slot = backward_[slot].recordNext) {
            if (forward_[slot].member == memberId) return slot;
            previous = slot;
        }
        return -1;
    }

    /// @brief Remove one live slot from both lists and recycle it.
    ///
    /// @details The record list is repaired through the supplied predecessor,
    /// the member list through its stored previous link. A record whose last
    /// dependency leaves stays registered without edges.
    /// @param slot Live slot to retire.
    /// @param recordPrevious Record-list predecessor from findOnRecord.
    /// @return Nothing.
    /// @invariant The retired slot enters the free list with revision zero.
    void unlink(int32_t slot, int32_t recordPrevious) {
        const ForwardWords node = forward_[slot];
        const BackwardWords links = backward_[slot];
        assert(node.revision > 0 && live_ > 0);
        if (recordPrevious >= 0) {
            BackwardWords previous = backward_[recordPrevious];
            assert(previous.recordNext == slot);
            previous.recordNext = links.recordNext;
            backward_.setAt(recordPrevious, previous);
        } else {
            assert(records_.valueAt(node.record) == slot);
            records_.setValueAt(node.record, links.recordNext);
        }
        if (links.memberPrevious >= 0) {
            ForwardWords previous = forward_[links.memberPrevious];
            assert(previous.memberNext == slot);
            previous.memberNext = node.memberNext;
            forward_.setAt(links.memberPrevious, previous);
        } else {
            assert(members_.valueAt(node.member) == slot);
            members_.setValueAt(node.member, node.memberNext);
        }
        if (node.memberNext >= 0) {
            BackwardWords next = backward_[node.memberNext];
            next.memberPrevious = links.memberPrevious;
            backward_.setAt(node.memberNext, next);
        }
        forward_.setAt(slot, ForwardWords{freeHead_, 0, 0, 0});
        backward_.setAt(slot, BackwardWords{-1, -1});
        freeHead_ = slot;
        --live_;
    }

public:
    /// @brief Bind empty records and dependencies to one arena.
    ///
    /// @details All payload and lookup pages remain on the owning logic
    /// block's arena; every mutation escalates the owner's dirty state.
    /// Construction allocates no record, token or slot.
    /// @param arena Resident owning arena, which outlives this store.
    /// @param dirty The owning index's raw-change state.
    /// @return An empty dependency store.
    CanonicalDependencyEdges(LbArena* arena, DirtyState* dirty)
        : members_(arena, dirty), records_(arena, dirty), forward_(arena, dirty), backward_(arena, dirty) {}

    /// @brief Register a record as live, keeping its identifier and edges.
    ///
    /// @details An absent record receives the next record identifier; a
    /// retired record becomes live again under its retained identifier; a live
    /// record is unchanged.
    /// @param record Exact packed record identity.
    /// @return The record's identifier, at least 1.
    int32_t markRecord(int64_t record) {
        HeadMap::InsertHint hint;
        const int32_t id = records_.lookupForInsert(record, hint);
        if (id == 0) return records_.insertAt(record, kNoEdge, hint);
        if (records_.valueAt(id) == kRetired) records_.setValueAt(id, kNoEdge);
        return id;
    }

    /// @brief Test whether a record is registered and not retired.
    ///
    /// @details Token-free records are live; this read changes nothing.
    /// @param record Exact packed record identity.
    /// @return True exactly for a live registered record.
    bool liveRecord(int64_t record) const {
        const int32_t* state = records_.find(record);
        return state != nullptr && *state != kRetired;
    }

    /// @brief Retire a live record whose dependencies are already removed.
    ///
    /// @details The record keeps its identifier until key compaction or
    /// release. An absent or already retired record is a defined no-op.
    /// @param record Exact packed record identity.
    /// @return Nothing.
    /// @invariant A live record retires only without remaining dependencies.
    void retireRecord(int64_t record) {
        const int32_t id = records_.lookup(record);
        if (id == 0 || records_.valueAt(id) == kRetired) return;
        assert(records_.valueAt(id) == kNoEdge && "a record retires only after its dependencies");
        records_.setValueAt(id, kRetired);
    }

    /// @brief Count registered records, live or retired.
    ///
    /// @details Identifiers run from 1 to this count in registration order,
    /// with key compaction preserving the survivors' relative order.
    /// @return Registered record count.
    int32_t recordCount() const { return records_.count(); }

    /// @brief Read the record at one identifier.
    ///
    /// @details Returns the exact packed identity without a borrowed reference.
    /// @param id Record identifier in [1, recordCount()].
    /// @return Packed record identity.
    int64_t recordAt(int32_t id) const { return records_.decodeKey(id); }

    /// @brief Test the record at one identifier for liveness.
    ///
    /// @details Equivalent to liveRecord on that identifier's key.
    /// @param id Record identifier in [1, recordCount()].
    /// @return True for a live record.
    bool liveRecordAt(int32_t id) const { return records_.valueAt(id) != kRetired; }

    /// @brief Insert a distinct dependency at both list heads.
    ///
    /// @details The record becomes live first. Duplicates leave the counter
    /// and every list unchanged. A retired slot can serve a different pair.
    /// Fresh edges receive the next caller-owned logical revision.
    /// @param token Exact packed dependency name and scope.
    /// @param record Exact packed owning record identity.
    /// @param revision Monotone logical revision, incremented only on insertion.
    /// @return True exactly when a new live edge is inserted.
    /// @invariant Revision overflow asserts before changing edge storage.
    bool insert(int64_t token, int64_t record, int32_t& revision) {
        const int32_t recordId = markRecord(record);
        HeadMap::InsertHint hint;
        int32_t memberId = members_.lookupForInsert(token, hint);
        int32_t previous = -1;
        if (memberId != 0 && findOnRecord(memberId, recordId, previous) >= 0) return false;
        assert(revision < INT32_MAX && "canonical dependency revision overflow");
        if (memberId == 0) memberId = members_.insertAt(token, kNoEdge, hint);
        const int32_t memberHead = members_.valueAt(memberId);
        const int32_t recordHead = records_.valueAt(recordId);
        assert(recordHead != kRetired);
        const ForwardWords node{memberHead, ++revision, memberId, recordId};
        const BackwardWords links{-1, recordHead};
        int32_t slot;
        if (freeHead_ >= 0) {
            slot = freeHead_;
            assert(forward_[slot].revision == 0);
            freeHead_ = forward_[slot].memberNext;
            forward_.setAt(slot, node);
            backward_.setAt(slot, links);
        } else {
            slot = forward_.size();
            assert(slot < INT32_MAX);
            forward_.push_back(node);
            backward_.push_back(links);
        }
        if (memberHead >= 0) {
            BackwardWords next = backward_[memberHead];
            next.memberPrevious = slot;
            backward_.setAt(memberHead, next);
        }
        members_.setValueAt(memberId, slot);
        records_.setValueAt(recordId, slot);
        ++live_;
        return true;
    }

    /// @brief Remove an exact dependency from both traversal directions.
    ///
    /// @details Missing pairs are defined no-ops. The removed slot enters the
    /// reusable free list; the record stays registered.
    /// @param token Packed dependency name and scope.
    /// @param record Packed owning record identity.
    /// @return True exactly when an existing live edge was removed.
    bool erase(int64_t token, int64_t record) {
        const int32_t memberId = members_.lookup(token);
        const int32_t recordId = records_.lookup(record);
        if (memberId == 0 || recordId == 0) return false;
        int32_t previous = -1;
        const int32_t slot = findOnRecord(memberId, recordId, previous);
        if (slot < 0) return false;
        unlink(slot, previous);
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
        const HeadMap& heads = ByMember ? members_ : records_;
        const int32_t id = heads.lookup(key);
        if (id == 0) return;
        int32_t slot = heads.valueAt(id);
        while (slot >= 0) {
            const ForwardWords& node = forward_[slot];
            assert(node.revision > 0 && (ByMember ? node.member : node.record) == id);
            const int32_t revision = node.revision;
            const int64_t other = ByMember ? records_.decodeKey(node.record) : members_.decodeKey(node.member);
            slot = ByMember ? node.memberNext : backward_[slot].recordNext;
            if (!sink(other, revision)) break;
        }
    }

    /// @brief Release slot storage once no dependency is live.
    ///
    /// @details Records keep their registration and liveness. Member
    /// identifiers are referenced only by slots, so the member map is released
    /// with them. A store with live dependencies keeps its free slots for reuse.
    /// @return Nothing.
    void maintain() {
        if (live_ != 0 || forward_.size() == 0) return;
        forward_.clear(); backward_.clear(); members_.resetToFresh();
        freeHead_ = -1;
    }

    /// @brief Remove selected edges and release storage when none remain.
    ///
    /// @details Selection sees the exact stored token and record, in slot
    /// order. Unlinking preserves surviving order and stable slots throughout
    /// selection. Records stay registered.
    /// @tparam Remove Pure predicate over token and record identities.
    /// @param remove True for an edge that should be retired.
    /// @return Nothing.
    template<class Remove>
    void eraseIf(Remove&& remove) {
        const int32_t count = forward_.size();
        for (int32_t slot = 0; slot < count; ++slot) {
            const ForwardWords node = forward_[slot];
            if (node.revision == 0) continue;
            if (!remove(members_.decodeKey(node.member), records_.decodeKey(node.record))) continue;
            int32_t previous = -1;
            const int32_t found = findOnRecord(node.member, node.record, previous);
            assert(found == slot);
            unlink(found, previous);
        }
        maintain();
    }

    /// @brief Drop dead record and token keys and renumber the survivors.
    ///
    /// @details Compaction keeps the survivors' relative order. Each surviving
    /// key's list is then walked once to store its new identifier, so every
    /// live slot is rewritten through exactly one record list and one member
    /// list; links and revisions are unchanged.
    /// @tparam Dead Pure predicate over packed record and token keys.
    /// @param dead True for a key that should leave its map.
    /// @return Nothing.
    /// @invariant Dead keys hold no live dependency: callers erase their edges first.
    template<class Dead>
    void eraseKeys(const Dead& dead) {
        const auto compact = [&](HeadMap& heads) {
            return heads.eraseIf([&](int64_t key) {
                if (!dead(key)) return false;
                const int32_t* head = heads.find(key);
                assert(head != nullptr && *head < 0 && "a dead key keeps no live dependency");
                return true;
            });
        };
        if (compact(records_) != 0) {
            for (int32_t id = 1; id <= records_.count(); ++id)
                for (int32_t slot = records_.valueAt(id); slot >= 0; slot = backward_[slot].recordNext) {
                    ForwardWords node = forward_[slot];
                    if (node.record == id) continue;
                    node.record = id;
                    forward_.setAt(slot, node);
                }
        }
        if (compact(members_) != 0) {
            for (int32_t id = 1; id <= members_.count(); ++id)
                for (int32_t slot = members_.valueAt(id); slot >= 0; slot = forward_[slot].memberNext) {
                    ForwardWords node = forward_[slot];
                    if (node.member == id) continue;
                    node.member = id;
                    forward_.setAt(slot, node);
                }
        }
    }

    /// @brief Release every record, dependency and derived storage.
    ///
    /// @details No free-slot or list-head reference survives teardown. The
    /// caller owns the logical revision counter and decides when to reset it.
    /// @return Nothing.
    void clear() {
        members_.resetToFresh(); records_.resetToFresh(); forward_.clear(); backward_.clear();
        freeHead_ = -1; live_ = 0;
    }

    /// @brief Count live edges, excluding free slots.
    ///
    /// @details This is a storage-accounting accessor with no proof-state effect.
    /// @return Current number of live exact dependencies.
    int32_t size() const { return live_; }

    /// @brief Account for the store's allocated arena pages.
    ///
    /// @details Includes both maps and both slot columns, including retained
    /// reusable capacity; it excludes the enclosing index's independent maps.
    /// @return Held storage bytes reported by the underlying cold containers.
    int64_t liveBytes() const {
        return peak_snapshot::measure("members_", members_, [&] { return members_.liveBytes(); })
            + peak_snapshot::measure("records_", records_, [&] { return records_.liveBytes(); })
            + peak_snapshot::measure("forward_", forward_, [&] { return forward_.liveBytes(); })
            + peak_snapshot::measure("backward_", backward_, [&] { return backward_.liveBytes(); });
    }
};
}
