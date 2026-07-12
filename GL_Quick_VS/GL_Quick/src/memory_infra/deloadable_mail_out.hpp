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

#include "cold_string_table.hpp"
#include "dirty_state.hpp"
#include "lb_arena.hpp"
#include "mail_types.hpp"
#include "typed_cold_map.hpp"

#include <cstdint>
#include <set>
#include <utility>

namespace gl {

    /// @brief Reserved deload tag-block base for `LbMemory::mailOut`.
    /// @details Eight facets occupy `base+0..7`; the append-only deload
    ///          directory reserves the full 50-tag block `755..804`.
    constexpr uint32_t kMailOutDeloadBase = 755u;

    /// @brief One LB's outgoing cross-LB mailbox, including its own string
    ///        interner, on the LB's deloadable arena.
    ///
    /// @details
    /// The phase-3 producer owns this mailbox while it owns the LB. Statements
    /// and origin history are stored with ids from `strings_`, a dedicated
    /// per-LB string table used by both columns. The complete unit participates
    /// in `LbMemory::visitContainers`: it therefore deloads with the LB after
    /// phase 3, reloads only for the serial commit barrier when `mailOutPending`
    /// says content exists, and returns its pages when the committed batch is
    /// cleared. `mailIn` remains a transient `RoutingColdMail` on the mail pool:
    /// phase 1 fills and absorbs it under the destination LB's worker claim,
    /// then its self-owned arena returns every block immediately.
    ///
    /// Eight facets form one append-only tag band: string lengths and bytes;
    /// statement-key lengths and bytes; origin keys, run starts, blob starts,
    /// and blob bytes. The mailbox deliberately has no disintegration-signal
    /// column because cross-LB mail serializes only statements and origins.
    ///
    /// @invariant All ids in `statements_` and `origins_`, including dependency
    ///            ids, belong to this instance's `strings_` table.
    ///            [I-163]
    /// @invariant The producing worker and the post-join commit barrier access
    ///            one LB's mailbox only while that LB is resident and claimed.
    ///            [I-163]
    /// @see RoutingColdMail, MailLog, LbMemory::visitContainers.
    struct DeloadableMailOut {
        ColdStringTable strings_;
        ColdStringTable::LengthsView stringsLengths;
        ColdStringTable::BytesView stringsBytes;
        TypedColdSet<IntMailStatementKey> statements_;
        TypedColdSet<IntMailStatementKey>::LengthsView statementsLengths;
        TypedColdSet<IntMailStatementKey>::BytesView statementsBytes;
        TypedColdBlobMap<int64_t, IntMailOrigin> origins_;
        TypedColdBlobMap<int64_t, IntMailOrigin>::KeysView originsKeys;
        TypedColdBlobMap<int64_t, IntMailOrigin>::RunStartsView originsRunStarts;
        TypedColdBlobMap<int64_t, IntMailOrigin>::BlobStartsView originsBlobStarts;
        TypedColdBlobMap<int64_t, IntMailOrigin>::BlobPoolView originsBlobPool;

        /// @brief Bind the dedicated string table and the two mail columns to
        ///        one LB's deloadable arena.
        ///
        /// @details Every facet shares the LB's real dirty state, so the normal
        /// deload writer observes all mailbox and interner mutations. The
        /// aggregate borrows both arguments and owns no arena.
        ///
        /// @param arena The owning `LbMemory::manager`.
        /// @param dirty The owning `LbMemory::dirty` state.
        DeloadableMailOut(LbArena* arena, DirtyState* dirty)
            : strings_(arena, dirty),
              stringsLengths(&strings_),
              stringsBytes(&strings_),
              statements_(arena, dirty),
              statementsLengths(&statements_.inner()),
              statementsBytes(&statements_.inner()),
              origins_(arena, dirty),
              originsKeys(&origins_.inner()),
              originsRunStarts(&origins_.inner()),
              originsBlobStarts(&origins_.inner()),
              originsBlobPool(&origins_.inner()) {}

        DeloadableMailOut(const DeloadableMailOut&) = delete;
        DeloadableMailOut& operator=(const DeloadableMailOut&) = delete;

        /// @brief Insert one statement using ids from this mailbox's interner.
        ///
        /// @details The whole `(expression id, validity id, levels)` record is
        /// minted into the statement set, preserving routing-mail set semantics.
        ///
        /// @param originalId Expression id in `strings_`.
        /// @param validityId Validity id in `strings_`.
        /// @param levels The statement's ascending level set.
        void insertStatement(int32_t originalId, int32_t validityId,
            const std::set<int>& levels) {
            IntMailStatementKey key;
            key.originalId = originalId;
            key.validityId = validityId;
            key.levels.assign(levels.begin(), levels.end());
            statements_.mint(key);
        }

        /// @brief Insert one statement from an ascending-unique level run.
        ///
        /// @details This is byte-identical to the set overload when the run is
        /// the ordered contents of that set, and avoids a transient set at the
        /// phase-3 mail-production door.
        ///
        /// @param originalId Expression id in `strings_`.
        /// @param validityId Validity id in `strings_`.
        /// @param levels Pointer to `levelCount` ascending-unique levels; null
        ///               only when `levelCount` is zero.
        /// @param levelCount Number of levels.
        void insertStatement(int32_t originalId, int32_t validityId,
            const int* levels, int32_t levelCount) {
            assert(levelCount >= 0 && (levelCount == 0 || levels != nullptr));
            IntMailStatementKey key;
            key.originalId = originalId;
            key.validityId = validityId;
            key.levels.assign(levels, levels + levelCount);
            statements_.mint(key);
        }

        /// @brief Whether the statements column is empty.
        /// @return `true` when no outgoing statement is stored.
        bool statementsEmpty() const { return statements_.count() == 0; }

        /// @brief Whether the origins column is empty.
        /// @return `true` when no outgoing origin row is stored.
        bool exprOriginMapEmpty() const { return origins_.count() == 0; }

        /// @brief Whether both outgoing-mail columns are empty.
        /// @return `true` when the mailbox carries no serialized content.
        bool empty() const {
            return statementsEmpty() && exprOriginMapEmpty();
        }

        /// @brief Clear the delivered mailbox and its private id space.
        ///
        /// @details All three stores reset together, so no surviving record can
        /// refer to a rebound string id. Pages return to the LB arena's free
        /// lists and may be reused by the rest of the resident LB immediately.
        void clear() {
            statements_.resetToFresh();
            origins_.resetToFresh();
            strings_.resetToFresh();
        }

        /// @brief Release all mailbox and interner page capacity.
        ///
        /// @details Used by full aggregate teardown. The owning `LbArena`
        /// remains responsible for returning its blocks to the main pool.
        void release() {
            statements_.release();
            origins_.release();
            strings_.release();
        }

        /// @brief Erase statements selected by a scope predicate.
        ///
        /// @details Origins and the private string table remain intact, matching
        /// the existing subtree-wipe contract for routing mail.
        ///
        /// @tparam Pred Callable accepting one `IntMailStatementKey`.
        /// @param pred Returns `true` for statement records to remove.
        /// @return Number of removed statements.
        template <typename Pred>
        int32_t filterStatements(Pred pred) {
            return statements_.eraseIf(
                [&pred](const IntMailStatementKey& key) { return pred(key); });
        }

        /// @brief Approximate live bytes in the private interner and columns.
        /// @return Sum of live bytes reported by all three cold stores.
        int64_t liveBytes() const {
            return strings_.liveBytes() + statements_.liveBytes()
                + origins_.liveBytes();
        }

        /// @brief Enumerate the eight deload facets from a reserved base tag.
        ///
        /// @details String facets precede every id-bearing column so reload
        /// restores the private id space before any consumer can decode it.
        ///
        /// @tparam Self `DeloadableMailOut` or its const form.
        /// @tparam Visitor Callable accepting `(uint32_t, facet&)`.
        /// @param self Mailbox instance to enumerate.
        /// @param base Reserved tag-band base.
        /// @param visit Facet visitor.
        template <typename Self, typename Visitor>
        static void visitContainersImpl(Self& self, uint32_t base,
            Visitor&& visit) {
            visit(base + 0u, self.stringsLengths);
            visit(base + 1u, self.stringsBytes);
            visit(base + 2u, self.statementsLengths);
            visit(base + 3u, self.statementsBytes);
            visit(base + 4u, self.originsKeys);
            visit(base + 5u, self.originsRunStarts);
            visit(base + 6u, self.originsBlobStarts);
            visit(base + 7u, self.originsBlobPool);
        }

        /// @brief Enumerate mutable deload facets.
        /// @tparam Visitor Callable accepting `(uint32_t, facet&)`.
        /// @param base Reserved tag-band base.
        /// @param visit Facet visitor.
        template <typename Visitor>
        void visitContainers(uint32_t base, Visitor&& visit) {
            visitContainersImpl(*this, base, std::forward<Visitor>(visit));
        }

        /// @brief Enumerate read-only deload facets.
        /// @tparam Visitor Callable accepting `(uint32_t, const facet&)`.
        /// @param base Reserved tag-band base.
        /// @param visit Facet visitor.
        template <typename Visitor>
        void visitContainers(uint32_t base, Visitor&& visit) const {
            visitContainersImpl(*this, base, std::forward<Visitor>(visit));
        }
    };

} // namespace gl
