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
#include "../memory.hpp"
namespace gl {
/// @brief The complete class runs one refresh rebuilt, held until the
        ///        refresh writes them to the class store in one pass.
        ///
        /// @details `refreshCanonicalClasses` rebuilds a scope and every
        /// initialized descendant. Splicing each rebuilt run into
        /// `canonicalEquivalenceClassesMap` moves the store's pool tail once per
        /// scope; instead `rebuildCanonicalScope` appends a changed run here and
        /// `flushCanonicalRuns` writes all held runs with one scratch rebuild of
        /// the store. A descendant rebuilt later in the same refresh reads its
        /// parent's new run from here. Record bytes, starts and lengths ride the
        /// page tier of the refresh's scratch arena, so they survive the
        /// per-scope scratch frames; the scope index is a fixed open-addressing
        /// table on the refresh's byte tier, sized for the refresh's scope count.
        /// @invariant Each scope is held at most once; records keep their order.
        /// @see rebuildCanonicalScope, flushCanonicalRuns, refreshCanonicalClasses.
        class CanonicalRunOverlay {
private:
            /// One held run: its scope and its records' first index and count.
            struct Entry { NameId scope; int32_t firstRecord; int32_t records; };

            DirtyState dirty = DirtyState::Clean;
            PagedVector<char> bytes;
            PagedVector<int32_t> starts;
            PagedVector<int32_t> lengths;
            PagedVector<Entry> entries;
            NameId* slotScope = nullptr;
            int32_t* slotEntry = nullptr;
            int32_t slotMask = 0;

public:
            /// @brief Create an empty overlay for one refresh.
            ///
            /// @details Binds the record columns to @p arena's page tier and
            /// allocates the scope index on its byte tier: a power-of-two table
            /// of at least twice @p scopeCapacity slots, all empty.
            /// @param arena The refresh's scratch arena; the index lives in the
            ///        caller's current scratch frame.
            /// @param scopeCapacity Upper bound on the scopes this refresh holds.
            /// @return (constructor)
            CanonicalRunOverlay(ScratchArena& arena, int32_t scopeCapacity)
                : bytes(&arena, &dirty), starts(&arena, &dirty),
                  lengths(&arena, &dirty), entries(&arena, &dirty) {
                assert(scopeCapacity >= 0);
                int32_t slots = 8;
                while (slots < 2 * scopeCapacity) slots *= 2;
                slotMask = slots - 1;
                slotScope = reinterpret_cast<NameId*>(arena.resolve(arena.alloc(
                    slots * static_cast<int32_t>(sizeof(NameId)), static_cast<int32_t>(alignof(NameId)))));
                slotEntry = reinterpret_cast<int32_t*>(arena.resolve(arena.alloc(
                    slots * static_cast<int32_t>(sizeof(int32_t)), static_cast<int32_t>(alignof(int32_t)))));
                for (int32_t s = 0; s < slots; ++s) { slotScope[s] = 0; slotEntry[s] = -1; }
            }
            CanonicalRunOverlay(const CanonicalRunOverlay&) = delete;
            CanonicalRunOverlay& operator=(const CanonicalRunOverlay&) = delete;

            /// @brief Hold one scope's complete new run.
            ///
            /// @details Copies the @p n records into the page-tier columns and
            /// indexes the scope. The records must be in final run order.
            /// @param scope Rebuilt scope; not yet held.
            /// @param runs The run's records.
            /// @param n Record count; `>= 0`.
            /// @return (void)
            void add(NameId scope, const StrSpan* runs, int32_t n) {
                assert(scope != 0 && n >= 0 && find(scope) < 0
                    && "CanonicalRunOverlay::add: scope held twice");
                assert(entries.size() * 2 < slotMask + 1
                    && "CanonicalRunOverlay::add: more scopes than the declared capacity");
                entries.push_back(Entry{ scope, lengths.size(), n });
                for (int32_t r = 0; r < n; ++r) {
                    starts.push_back(bytes.size());
                    lengths.push_back(static_cast<int32_t>(runs[r].len));
                    if (runs[r].len > 0) bytes.appendRun(runs[r].ptr, static_cast<int32_t>(runs[r].len));
                }
                uint32_t slot = (static_cast<uint32_t>(scope) * 2654435761u) & static_cast<uint32_t>(slotMask);
                while (slotScope[slot] != 0) slot = (slot + 1) & static_cast<uint32_t>(slotMask);
                slotScope[slot] = scope;
                slotEntry[slot] = entries.size() - 1;
            }

            /// @brief Index of the held run of @p scope.
            ///
            /// @details A scope whose run this refresh did not change is not
            /// held; its current run is the one in the class store.
            /// @param scope Any scope id.
            /// @return The entry index, or `-1` when the scope is not held.
            int32_t find(NameId scope) const {
                uint32_t slot = (static_cast<uint32_t>(scope) * 2654435761u) & static_cast<uint32_t>(slotMask);
                while (slotScope[slot] != 0) {
                    if (slotScope[slot] == scope) return slotEntry[slot];
                    slot = (slot + 1) & static_cast<uint32_t>(slotMask);
                }
                return -1;
            }

            /// @brief Copy one held record into contiguous bytes.
            ///
            /// @details Allocates on @p arena's byte tier in the caller's
            /// current frame; the record may straddle pages in the overlay.
            /// @param entry Entry index from `find`.
            /// @param r Record position in `[0, entries[entry].records)`.
            /// @param arena Destination arena.
            /// @return The record's bytes.
            StrSpan copyRecord(int32_t entry, int32_t r, ScratchArena& arena) const {
                const Entry e = entries[entry];
                assert(r >= 0 && r < e.records);
                const int32_t len = lengths[e.firstRecord + r];
                char* out = arena.resolve(arena.alloc(len > 0 ? len : 1, 1));
                if (len > 0) bytes.copyRunTo(starts[e.firstRecord + r], len, out);
                return StrSpan(out, len);
            }
        
    /// @brief Number of held scope runs.
    /// @details Reads the existing entry column without allocating or copying records.
    /// @return Number of runs, including runs containing zero records.
    int32_t entryCount() const { return entries.size(); }

    /// @brief Scope of one held run.
    /// @details Uses the existing entry-column bounds contract.
    /// @param entry Held-run index from find or the entryCount range.
    /// @return The held scope identifier.
    NameId scopeAt(int32_t entry) const { return entries[entry].scope; }

    /// @brief Number of records in one held run.
    /// @details Uses the existing entry-column bounds contract.
    /// @param entry Held-run index from find or the entryCount range.
    /// @return Number of ordered records in the run.
    int32_t recordCount(int32_t entry) const { return entries[entry].records; }
};
}
