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
#include "scratch_arena.hpp"

#include <cassert>
#include <cstdint>

namespace gl {

    /// @brief One ordered OR leaf and its fully instantiated complement.
    /// @details Both ids belong to the owning block's NameMap; their exact
    /// spelling is preserved for proof history while matching uses a view.
    struct OrImplicationLeaf {
        NameId literalId;
        NameId complementId;
    };

    /// @brief One OR source's immutable payload locations and processing flags.
    /// @details The map key supplies the original OR id and source validity.
    /// Payload offsets are element offsets in paged columns, never addresses.
    struct OrImplicationSource {
        int32_t leafStart;
        int32_t leafCount;
        int32_t levelStart;
        int32_t levelCount;
        uint32_t flags;
        NameId canonicalId;
    };

    /// @brief Canonical-deloadable source state for OR implications on demand.
    ///
    /// @details
    /// A source is stored once under its original expression/validity pair.
    /// Leaf and level runs are append-built paged columns. Retirement makes a
    /// source unavailable immediately, then a batched drain erases its record
    /// and compacts payload columns when dead payload exceeds live payload.
    /// Pending events are logical state and survive eviction with the sources.
    /// Derived matching indexes are separate from this canonical source store.
    /// @invariant No stored pointer refers to a payload or interner buffer.
    /// @see OrImplicationPostings
    struct OrImplications {
        static constexpr uint32_t Active = 1u;
        static constexpr uint32_t Local = 2u;
        static constexpr uint32_t kFacetCount = 8u;

        TypedColdMap<int64_t, OrImplicationSource> sources;
        TypedColdMap<int64_t, OrImplicationSource>::KeysView sourceKeys;
        TypedColdMap<int64_t, OrImplicationSource>::ValuesView sourceValues;
        PagedVector<OrImplicationLeaf> leaves;
        PagedVector<int> levels;
        PagedVector<int64_t> pendingSources;
        PagedVector<int64_t> pendingFacts;
        PagedVector<int64_t> pendingClasses;
        PagedVector<int64_t> retiredSources;

        /// @brief Bind empty source columns to the owning block's arena.
        /// @details Every persisted mutation participates in the block's dirty
        /// state; canonical deload visits these columns in fixed facet order.
        /// @param arena The owning block or test arena.
        /// @param dirty The aggregate content-change state.
        /// @return A constructed empty source store.
        OrImplications(LbArena* arena, DirtyState* dirty)
            : sources(arena, dirty), sourceKeys(&sources.inner()),
              sourceValues(&sources.inner()), leaves(arena, dirty),
              levels(arena, dirty), pendingSources(arena, dirty),
              pendingFacts(arena, dirty), pendingClasses(arena, dirty),
              retiredSources(arena, dirty) {}

        /// @brief Record one admitted OR source and stage its first evaluation.
        /// @details Exact duplicate source identities retain their first record.
        /// A retired source cannot be resurrected before its retirement drain.
        /// Inputs are copied before return, including their original leaf order.
        /// @param key Packed original OR id and source-validity id.
        /// @param canonicalId The canonical OR identity at its source scope.
        /// @param leafRun Ordered true-polarity leaf/complement pairs.
        /// @param leafCount Number of leaves, at least two.
        /// @param levelRun Source derivation levels, ascending and unique.
        /// @param levelCount Number of supplied levels, zero for a neutral run.
        /// @param local Whether this block owns the source's deduction work.
        /// @return True exactly when a new source was recorded.
        bool insert(int64_t key, NameId canonicalId,
                    const OrImplicationLeaf* leafRun, int32_t leafCount,
                    const int* levelRun, int32_t levelCount, bool local) {
            assert(key != 0 && canonicalId != 0 && leafRun != nullptr);
            assert(leafCount >= 2 && levelCount >= 0);
            assert(levelCount == 0 || levelRun != nullptr);
            if (sources.lookup(key) != 0) return false;
            const OrImplicationSource source{
                leaves.size(), leafCount, levels.size(), levelCount,
                Active | (local ? Local : 0u), canonicalId};
            for (int32_t i = 0; i < leafCount; ++i) {
                assert(leafRun[i].literalId != 0 && leafRun[i].complementId != 0);
                leaves.push_back(leafRun[i]);
            }
            for (int32_t i = 0; i < levelCount; ++i) levels.push_back(levelRun[i]);
            sources.insert(key, source);
            pendingSources.push_back(key);
            return true;
        }

        /// @brief Find an active source by its stable logical identity.
        /// @details An absent or retired source is a defined miss. The returned
        /// pointer remains valid only until a mutation of the source table.
        /// @param key Packed source expression and validity.
        /// @return The active record, or nullptr when no active source exists.
        const OrImplicationSource* find(int64_t key) const {
            const OrImplicationSource* source = sources.find(key);
            return source != nullptr && (source->flags & Active) != 0
                ? source : nullptr;
        }

        /// @brief Retire one source before unlinking its derived postings.
        /// @details Repeated retirement is a defined no-op. A source becomes
        /// invisible to every active-source probe before any table compaction.
        /// @param key Packed source expression and validity.
        /// @return True exactly when an active source was retired.
        bool retire(int64_t key) {
            const OrImplicationSource* found = find(key);
            if (found == nullptr) return false;
            OrImplicationSource source = *found;
            source.flags &= ~Active;
            sources.upsert(key, source);
            retiredSources.push_back(key);
            return true;
        }

        /// @brief Remove retired records and bound retained payload storage.
        /// @details Called once after all retired sources' derived postings have
        /// been unlinked. Surviving keys retain order; payload locations change
        /// only when dead leaf/level elements exceed the corresponding live run.
        /// Pending source events use stable keys and remain safe across erasure.
        /// @param scratch A page-tier scratch arena distinct from the source arena.
        /// @return Void.
        void compactRetired(ScratchArena& scratch) {
            if (retiredSources.size() == 0) return;
            int32_t liveLeaves = 0;
            int32_t liveLevels = 0;
            sources.eraseIf([&](int64_t key) {
                const OrImplicationSource* source = sources.find(key);
                assert(source != nullptr);
                if ((source->flags & Active) == 0) return true;
                liveLeaves += source->leafCount;
                liveLevels += source->levelCount;
                return false;
            });
            retiredSources.clear();
            if (sources.empty()) {
                leaves.clear();
                levels.clear();
                return;
            }
            if (leaves.size() - liveLeaves <= liveLeaves
                && levels.size() - liveLevels <= liveLevels) return;

            DirtyState temporaryDirty = DirtyState::Clean;
            PagedVector<OrImplicationLeaf> keptLeaves(&scratch, &temporaryDirty);
            PagedVector<int> keptLevels(&scratch, &temporaryDirty);
            for (int32_t id = 1; id <= sources.count(); ++id) {
                const int64_t key = sources.keyAt(id);
                OrImplicationSource source = sources.valueAt(id);
                const int32_t leafStart = keptLeaves.size();
                const int32_t levelStart = keptLevels.size();
                for (int32_t i = 0; i < source.leafCount; ++i)
                    keptLeaves.push_back(leaves[source.leafStart + i]);
                for (int32_t i = 0; i < source.levelCount; ++i)
                    keptLevels.push_back(levels[source.levelStart + i]);
                source.leafStart = leafStart;
                source.levelStart = levelStart;
                sources.upsert(key, source);
            }
            leaves = keptLeaves;
            levels = keptLevels;
        }

        /// @brief Deep-copy all source and pending-event state across arenas.
        /// @details Derived matching indexes are rebuilt by the receiving block.
        /// @param other The resident source store to copy.
        /// @return Void.
        void copyFrom(const OrImplications& other) {
            sources.copyFrom(other.sources);
            leaves = other.leaves;
            levels = other.levels;
            pendingSources = other.pendingSources;
            pendingFacts = other.pendingFacts;
            pendingClasses = other.pendingClasses;
            retiredSources = other.retiredSources;
        }

        /// @brief Enumerate writable canonical deload facets in fixed order.
        /// @details The caller supplies a non-overlapping append-only tag band.
        /// @tparam Visitor Callable receiving a uint32_t tag and a column.
        /// @param base First tag assigned to this source store.
        /// @param visit The synchronous deload/lifecycle visitor.
        /// @return Void.
        template<class Visitor>
        void visitContainers(uint32_t base, Visitor&& visit) {
            visit(base, sourceKeys);
            visit(base + 1, sourceValues);
            visit(base + 2, leaves);
            visit(base + 3, levels);
            visit(base + 4, pendingSources);
            visit(base + 5, pendingFacts);
            visit(base + 6, pendingClasses);
            visit(base + 7, retiredSources);
        }

        /// @brief Enumerate read-only canonical deload facets in fixed order.
        /// @details The facet order and tags match the writable overload.
        /// @tparam Visitor Callable receiving a uint32_t tag and a const column.
        /// @param base First tag assigned to this source store.
        /// @param visit The synchronous read-only visitor.
        /// @return Void.
        template<class Visitor>
        void visitContainers(uint32_t base, Visitor&& visit) const {
            visit(base, sourceKeys);
            visit(base + 1, sourceValues);
            visit(base + 2, leaves);
            visit(base + 3, levels);
            visit(base + 4, pendingSources);
            visit(base + 5, pendingFacts);
            visit(base + 6, pendingClasses);
            visit(base + 7, retiredSources);
        }

        /// @brief Release all logical state at block teardown.
        /// @details Pending arrivals and retirements disappear with their source
        /// records. Callers clear derived indexes before releasing arena pages.
        /// @return Void.
        void clear() {
            sources.resetToFresh();
            leaves.clear();
            levels.clear();
            pendingSources.clear();
            pendingFacts.clear();
            pendingClasses.clear();
            retiredSources.clear();
        }

        /// @brief Count all retained canonical source bytes.
        /// @details Includes unreclaimed payload runs and pending event columns.
        /// @return Pool-backed bytes retained by this source store.
        int64_t liveBytes() const {
            return peak_snapshot::measure("sources", sources, [&] { return sources.liveBytes(); }) + peak_snapshot::measure("leaves", leaves, [&] { return leaves.liveBytes(); }) + peak_snapshot::measure("levels", levels, [&] { return levels.liveBytes(); })
                + peak_snapshot::measure("pendingSources", pendingSources, [&] { return pendingSources.liveBytes(); }) + peak_snapshot::measure("pendingFacts", pendingFacts, [&] { return pendingFacts.liveBytes(); })
                + peak_snapshot::measure("pendingClasses", pendingClasses, [&] { return pendingClasses.liveBytes(); }) + peak_snapshot::measure("retiredSources", retiredSources, [&] { return retiredSources.liveBytes(); });
        }
    };
}
