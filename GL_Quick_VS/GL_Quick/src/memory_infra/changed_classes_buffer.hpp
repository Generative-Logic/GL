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

// ChangedClassesBuffer -- the per-step delta of changed equivalence classes,
// on the LB's cold DELOADABLE arena. push()/classAt() are declared here and
// defined out-of-line in memory.hpp (they touch EquivalenceClass + its
// serialize/deserialize, which are complete only there).

#include "dirty_state.hpp"
#include "lb_arena.hpp"
#include "paged_vector.hpp"
#include "scratch_arena.hpp"

#include <cstdint>
#include <vector>

namespace gl {

    struct EquivalenceClass;  // defined in memory.hpp; push/classAt are declared
                              // here and defined out-of-line there.
    struct EquivalenceClassView;  // defined in memory.hpp; classViewAt returns it
                                  // by value (defined out-of-line there).

    /// @brief Reserved deload tag-block base for `LbMemory::changedClassesThisStep`.
    /// @details 3 paged facets at `base+0..2` (validity ids, blob starts, blob
    ///          pool); a 50-tag block is reserved like the mail bands.
    constexpr uint32_t kChangedClassesDeloadBase = 655u;

    /// @brief The per-step delta of changed equivalence classes, on the LB's
    ///        cold DELOADABLE arena.
    ///
    /// @details
    /// Replaces the heap `std::vector<std::pair<NameId, EquivalenceClass>>`.
    /// `changedClassesThisStep` is written by `updateEquivalenceClasses` at
    /// the single-threaded seams — `standardProcessing` phase-1/3 (phase 2 is
    /// read-only on the LB, I-66/I-74), LB seeding, and the post-join barrier
    /// deposits — and cleared ONLY at `standardProcessing`'s tail (Step 7),
    /// after its consumers have read it. A seeding- or barrier-minted delta
    /// therefore survives across bursts (and across a deload / reload — the
    /// paged columns deload like any cold container) until the LB's next
    /// `standardProcessing` call consumes it. It is **dischargeable** (not
    /// in `survivesDischarge`): a discharged LB never applies classes again
    /// (I-112), so a pending delta dies with it by design.
    /// It rides the LB's per-LB `LbArena`
    /// (`LbMemory::manager`, passed in — NOT owned) with the LB's REAL
    /// deload-`dirty`, and stores its entries as parallel paged columns: one
    /// validity id per entry, plus a blob CSR (`blobStarts_` + `blobPool_`)
    /// holding each class's canonical serialization.
    ///
    /// `clear()` frees the columns' pages back to the arena's free-list (the
    /// per-step clear); `release()` drops the column page capacity at grid
    /// teardown (the arena itself is `releaseAll`ed by `LbMemory`). Enumerated by
    /// `LbMemory::visitContainers` at the reserved base so it deloads / reloads /
    /// releases like any other cold container.
    ///
    /// @invariant Single-threaded seam access only (`standardProcessing`
    ///            phase-1/3, LB seeding, post-join barrier deposits); may hold
    ///            pending deltas across deload / compaction boundaries.
    /// @see serializeEquivalenceClass, ColdMail, `PagedVector`.
    struct ChangedClassesBuffer {
        /// @brief One validity id per delta entry.
        PagedVector<NameId> validityIds_;
        /// @brief Per-entry blob byte offset (CSR over `blobPool_`).
        PagedVector<int32_t> blobStarts_;
        /// @brief Concatenated canonical class blobs, dense.
        PagedVector<char> blobPool_;

        /// @brief Bind the paged columns to the LB's deloadable arena + dirty.
        ///
        /// @param arena The LB's deloadable per-LB `LbArena` (`LbMemory::manager`).
        /// @param dirty The LB's deload-dirty flag (`LbMemory::dirty`).
        ChangedClassesBuffer(LbArena* arena, DirtyState* dirty)
            : validityIds_(arena, dirty),
              blobStarts_(arena, dirty),
              blobPool_(arena, dirty) {}
        ChangedClassesBuffer(const ChangedClassesBuffer&) = delete;
        ChangedClassesBuffer& operator=(const ChangedClassesBuffer&) = delete;

        /// @brief Append one (validityId, class snapshot) delta entry.
        ///
        /// @details Defined out-of-line in `memory.hpp` (uses
        /// `serializeEquivalenceClass` + `EquivalenceClass`).
        ///
        /// @param vid The class's validity id.
        /// @param cls The merged class to snapshot (serialized to a blob).
        void push(NameId vid, const EquivalenceClass& cls);

        /// @brief Append one (validityId, ALREADY-serialized blob) delta entry —
        ///        the heap-free twin used when the producer already holds the
        ///        canonical blob bytes (e.g. `MergeClassAccum::serialize()`),
        ///        so no `EquivalenceClass` needs materializing.
        ///
        /// @details The stored bytes are IDENTICAL to the `EquivalenceClass`
        /// overload's (both are the canonical `serializeEquivalenceClass`
        /// stream), so `classAt` / the deload image are byte-for-byte unchanged.
        /// Header-inline (no `EquivalenceClass` dependency).
        ///
        /// @param vid      The class's validity id.
        /// @param blobData The canonical class blob bytes.
        /// @param blobLen  The blob length (>= 0).
        void push(NameId vid, const char* blobData, int32_t blobLen) {
            validityIds_.push_back(vid);
            blobStarts_.push_back(blobPool_.size());
            if (blobLen > 0) blobPool_.appendRun(blobData, blobLen);
        }

        /// @brief Number of delta entries.
        /// @return Entry count (0 before the first push).
        int32_t size() const { return validityIds_.size(); }

        /// @brief Validity id of entry `i`.
        /// @param i Index in `[0, size())`.
        /// @return The entry's validity id.
        NameId validityAt(int32_t i) const { return validityIds_[i]; }

        /// @brief Decode entry `i`'s class snapshot (page-aware).
        ///
        /// @details Defined out-of-line in `memory.hpp` (uses
        /// `deserializeEquivalenceClass` + `EquivalenceClass`).
        ///
        /// @param i Index in `[0, size())`.
        /// @return The reconstructed class.
        EquivalenceClass classAt(int32_t i) const;

        /// @brief Zero-copy view over entry `i`'s class snapshot — the heap-free
        ///        `classAt` twin for the equi-class apply read path (equi-7).
        ///
        /// @details Defined out-of-line in `memory.hpp` (returns
        /// `EquivalenceClassView`). Contiguous blobs point into the pool; a
        /// straddling blob copies once into @p scratch.
        ///
        /// @param i       Index in `[0, size())`.
        /// @param scratch Caller reuse buffer, written only on a page straddle.
        /// @return A view over the entry's canonical class blob.
        EquivalenceClassView classViewAt(int32_t i,
                                         std::vector<char>& scratch) const;

        /// @brief Zero-copy view over entry `i`'s class snapshot — the
        ///        arena-backed twin of the `std::vector<char>` `classViewAt`,
        ///        assembling a page-straddling blob onto @p scratch's byte-bump
        ///        tier (equi-7 read path).
        ///
        /// @details The recursion-safe read door for the equivalence-class apply
        /// (`applyEquiClasses`). The contiguous common case returns a pointer
        /// straight into the paged `blobPool_` and never touches @p scratch; only
        /// a page-straddling blob is assembled — once — into a FRESH
        /// `scratch.alloc(len, 1)` byte-bump run. Because each straddle takes a
        /// fresh allocation with NO rewind, an outer view's bytes survive a nested
        /// re-entrant apply that itself peeks; the straddle buffers only
        /// accumulate until the per-task `releaseAll` reclaims the arena. The
        /// returned bytes are byte-identical to the `std::vector<char>` overload's.
        /// Defined out-of-line in `memory.hpp` (returns `EquivalenceClassView`).
        ///
        /// @param i       Index in `[0, size())`.
        /// @param scratch Byte-bump arena the straddle copy is assembled onto;
        ///                left untouched on the contiguous case.
        /// @return A view over the entry's canonical class blob.
        /// @see classViewAt(int32_t, std::vector<char>&) const — the vector twin;
        ///      `HashMap::peekBlobContiguous` — the sibling arena peek overload.
        EquivalenceClassView classViewAt(int32_t i,
                                         ScratchArena& scratch) const;

        /// @brief Empty the delta (frees the columns' pages to the arena
        ///        free-list) — the per-step clear.
        void clear() {
            validityIds_.clear();
            blobStarts_.clear();
            blobPool_.clear();
        }

        /// @brief Drop the columns' page capacity — the grid-teardown release.
        void release() {
            validityIds_.release();
            blobStarts_.release();
            blobPool_.release();
        }

        /// @brief Approximate live byte footprint across the three columns.
        /// @return Live bytes the buffer occupies on the arena.
        int64_t liveBytes() const {
            return validityIds_.liveBytes() + blobStarts_.liveBytes()
                + blobPool_.liveBytes();
        }

        /// @brief Enumerate the deload facets at `base + 0..2` — the seam
        ///        `LbMemory::visitContainers` splices the instance in.
        ///
        /// @details Three paged columns visited directly (like `validityNodes`),
        /// ascending: validity ids (`base+0`), blob starts (`base+1`), blob pool
        /// (`base+2`).
        ///
        /// @tparam Self    `ChangedClassesBuffer` or `const ChangedClassesBuffer`.
        /// @tparam Visitor Callable `(uint32_t tag, column&)`.
        /// @param self  The instance being enumerated.
        /// @param base  The instance's reserved tag-block base.
        /// @param visit The per-facet visitor.
        template <typename Self, typename Visitor>
        static void visitContainersImpl(Self& self, uint32_t base,
            Visitor&& visit) {
            visit(base + 0u, self.validityIds_);
            visit(base + 1u, self.blobStarts_);
            visit(base + 2u, self.blobPool_);
        }

        /// @brief Mutable facet enumeration (load / release / discharge walk).
        /// @tparam Visitor Callable `(uint32_t tag, column&)`.
        /// @param base  The instance's reserved tag-block base.
        /// @param visit The per-facet visitor.
        template <typename Visitor>
        void visitContainers(uint32_t base, Visitor&& visit) {
            visitContainersImpl(*this, base, std::forward<Visitor>(visit));
        }

        /// @brief Const facet enumeration (the deload DUMP reads a const LB).
        /// @tparam Visitor Callable `(uint32_t tag, const column&)`.
        /// @param base  The instance's reserved tag-block base.
        /// @param visit The per-facet visitor.
        template <typename Visitor>
        void visitContainers(uint32_t base, Visitor&& visit) const {
            visitContainersImpl(*this, base, std::forward<Visitor>(visit));
        }
    };

} // namespace gl
