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

/// @brief Index projected packed keys without changing their owning map.
/// @details Append-only key gains extend the index. A count/insert-epoch
/// difference detects erasure or reset and rebuilds it before querying.
/// Canonical replacement clears this derived object at the owning lifecycle seam.
/// Labels and postings use the owning arena and have no canonical deload facets.
class ExistenceKeyLookup {
    DirtyState dirty_ = DirtyState::Clean;
    int32_t scanned_ = 0;
    uint32_t epoch_ = 0;
    bool built_ = false;
public:
    ColdHashSet<BytesKeyStore> labels;
    OrPostingIndex<int64_t> postings;

    /// @brief Bind an empty derived lookup to its owning arena.
    /// @details No logical keys are created by construction.
    /// @param arena Resident owning arena, which outlives this object.
    /// @return An empty lookup.
    explicit ExistenceKeyLookup(LbArena* arena)
        : labels(arena, &dirty_), postings(arena) {}

    /// @brief Release every derived label and posting.
    /// @details Call before canonical release or replacement, while resident.
    /// @return Void.
    void clear() {
        postings.clear();
        labels.resetToFresh();
        scanned_ = 0;
        epoch_ = 0;
        built_ = false;
    }

    /// @brief Extend or rebuild the projection of the current key set.
    /// @details The projector is called only for keys not yet indexed. It must
    /// not mutate the source map; it may intern labels in this derived lookup.
    /// Erasure plus insertion at the same count also forces a rebuild.
    /// @tparam Map A typed cold map or set with packed int64_t keys.
    /// @tparam Project A callback mapping a packed key to its lookup label.
    /// @param map Authoritative resident map.
    /// @param project Pure key projection apart from derived label allocation.
    /// @return Void.
    template<class Map, class Project>
    void ensureCurrent(const Map& map, Project&& project) {
        const int32_t count = map.count();
        const uint32_t epoch = map.inner().insertEpoch();
        if (!built_ || count < scanned_
            || epoch - epoch_ != static_cast<uint32_t>(count - scanned_))
            clear();
        for (int32_t id = scanned_ + 1; id <= count; ++id) {
            const int64_t key = map.keyAt(id);
            postings.insert(project(key), key);
        }
        scanned_ = count;
        epoch_ = epoch;
        built_ = true;
    }

    /// @brief Visit the indexed keys for one exact projection.
    /// @details The lookup must first be synchronized with its source map.
    /// The callback must not mutate this lookup.
    /// @tparam Sink Callback receiving one packed map key.
    /// @param key Projected key from this lookup's identifier space.
    /// @param sink Consumer of matching packed keys.
    /// @return Void.
    template<class Sink>
    void forEach(int64_t key, Sink&& sink) const {
        assert(built_);
        postings.forEach(key, sink);
    }

    /// @brief Account for all derived storage in arena compaction.
    /// @details Includes normalized labels and posting structures.
    /// @return Live bytes owned by the lookup.
    int64_t liveBytes() const { return peak_snapshot::measure("labels", labels, [&] { return labels.liveBytes(); }) + peak_snapshot::measure("postings", postings, [&] { return postings.liveBytes(); }); }
};

/// @brief Index source/scope identities to the parked keys retaining them.
/// @details Built lazily from authoritative source records. Production insertion,
/// equality replacement and retirement update edges without scanning unrelated
/// sources. Multiple provenance rows of one source share a distinct edge.
class ExistenceSourceLookup {
public:
    OrPostingIndex<int64_t> postings;
    OrPostingIndex<int64_t> owners;
    bool built = false;

    /// @brief Bind an empty source index to its owning arena.
    /// @details The constructor creates no source or proof state.
    /// @param arena Resident owning arena.
    /// @return An unbuilt lookup.
    explicit ExistenceSourceLookup(LbArena* arena) : postings(arena), owners(arena) {}

    /// @brief Add one source identity's ownership of a parked key.
    /// @details Forward and reverse edges are distinct sets and update together.
    /// @param source Canonical source/scope key.
    /// @param park Stable marked-expression/scope key.
    /// @return Nothing.
    void add(int64_t source, int64_t park) {
        postings.insert(source, park);
        owners.insert(park, source);
    }

    /// @brief Remove one exact source/park edge from both directions.
    /// @details An absent edge is an ordinary no-op during duplicate retirement.
    /// @param source Previously indexed canonical source/scope key.
    /// @param park Stable parked key.
    /// @return Nothing.
    void remove(int64_t source, int64_t park) {
        postings.erase(source, park);
        owners.erase(park, source);
    }

    /// @brief Unlink only the old source identities owned by one parked key.
    /// @details Snapshot before mutation so other parks retain their index edges.
    /// @param park Stable parked key being replaced or retired.
    /// @param arena Caller-owned page-tier scratch.
    /// @return Nothing.
    void eraseOwner(int64_t park, LbArena& arena) {
        DirtyState dirty = DirtyState::Clean;
        PagedVector<int64_t> sources(&arena, &dirty);
        owners.forEach(park, [&](int64_t source) { sources.push_back(source); });
        for (int32_t i = 0; i < sources.size(); ++i) remove(sources[i], park);
    }

    /// @brief Release derived source edges and require reconstruction.
    /// @details The authoritative rejected maps remain unchanged.
    /// @return Void.
    void clear() { postings.clear(); owners.clear(); built = false; }

    /// @brief Account for source-index storage.
    /// @details This storage participates in arena compaction only.
    /// @return Live derived bytes.
    int64_t liveBytes() const { return peak_snapshot::measure("postings", postings, [&] { return postings.liveBytes(); }) + peak_snapshot::measure("owners", owners, [&] { return owners.liveBytes(); }); }
};

/// @brief Own the derived lookups used by existence replay and retirement.
/// @details These caches never persist expanded products or witness names.
/// Canonical release, scope wipe and discharge clear them; raw eviction retains
/// their pages and descriptors. Fresh clones reconstruct from canonical sources.
struct ExistenceLookupIndexes {
    ExistenceKeyLookup algebraDemand, integrationDemand, integrationSet, receipts;
    ExistenceSourceLookup algebraSources, integrationSources;

    /// @brief Bind all lookups to one logic block's arena.
    /// @details Construction allocates no logical state or canonical facets.
    /// @param arena Resident owning arena.
    /// @return Empty derived lookups.
    explicit ExistenceLookupIndexes(LbArena* arena)
        : algebraDemand(arena), integrationDemand(arena), integrationSet(arena),
          receipts(arena), algebraSources(arena), integrationSources(arena) {}

    /// @brief Clear all derived state at a logical or canonical lifecycle seam.
    /// @details Call while the arena is resident and before releasing its pages.
    /// @return Void.
    void clear() {
        algebraDemand.clear(); integrationDemand.clear(); integrationSet.clear();
        receipts.clear(); algebraSources.clear(); integrationSources.clear();
    }

    /// @brief Include every lookup in the owning arena's live-byte accounting.
    /// @details The canonical source-record footprint remains separately measured.
    /// @return Total live bytes of the six lookups.
    int64_t liveBytes() const {
        return peak_snapshot::measure("algebraDemand", algebraDemand, [&] { return algebraDemand.liveBytes(); }) + peak_snapshot::measure("integrationDemand", integrationDemand, [&] { return integrationDemand.liveBytes(); })
            + peak_snapshot::measure("integrationSet", integrationSet, [&] { return integrationSet.liveBytes(); }) + peak_snapshot::measure("receipts", receipts, [&] { return receipts.liveBytes(); })
            + peak_snapshot::measure("algebraSources", algebraSources, [&] { return algebraSources.liveBytes(); }) + peak_snapshot::measure("integrationSources", integrationSources, [&] { return integrationSources.liveBytes(); });
    }
};

} // namespace gl
