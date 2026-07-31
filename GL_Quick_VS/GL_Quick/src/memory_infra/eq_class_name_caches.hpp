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

// EqClassNameCaches -- the per-NameMap-id classifyName / token-scan memo, on
// the LB's cold DELOADABLE arena. kindOf()/tokensViewOf() are declared here
// and defined out-of-line in memory.hpp (they touch NameMap + classifyName +
// the occurrence scanners + SpecialTokenScanView, complete only there).

#include "dirty_state.hpp"
#include "lb_arena.hpp"
#include "paged_vector.hpp"
#include "typed_cold_map.hpp"

#include <cstdint>

namespace gl {

    struct NameMap;                // defined in memory.hpp
    struct SpecialTokenScan;       // defined in memory.hpp
    struct SpecialTokenScanView;   // defined in memory.hpp
    enum class NameKind : uint8_t; // defined in memory.hpp

    /// @brief Reserved deload tag-block base for `LbMemory::eqClassNameCaches`.
    /// @details 5 paged facets at `base+0..4` (kind-by-id, then the token blob
    ///          map's key + run-starts + blob-starts + blob-pool); a 50-tag block
    ///          is reserved like the mail / changed-class bands.
    constexpr uint32_t kEqClassNameCachesDeloadBase = 705u;

    /// @brief Lazy per-NameMap-id memoization of `classifyName` /
    ///        `scanSpecialTokens`, on the LB's cold DELOADABLE arena.
    ///
    /// @details
    /// `filterIterations` historically regex-scanned every class member and the
    /// whole statement text on every (statement × class) probe. Both scans are
    /// pure functions of the decoded string, and a NameMap id never re-binds to a
    /// different string while that NameMap lives — so both results are memoizable
    /// per id with no invalidation. `kindOf` backs the class-member bucketing;
    /// `tokensViewOf` backs the statement-side scan.
    ///
    /// The memo is a derived structure (a pure function of `nameMap`), but it
    /// rides the LB's per-LB `LbArena` (`LbMemory::manager`, passed in — NOT
    /// owned) with the LB's REAL deload-`dirty`, enrolled in
    /// `LbMemory::visitContainers` at the reserved base. It is **persistent**
    /// (never cleared per step) and `survivesDischarge`, so a discharged LB keeps
    /// it; on deload it round-trips with the LB (a future memory option is to
    /// drop-and-rebuild it instead, since it is reconstructible from `nameMap`).
    /// Two paged columns: `kindById_` (one `NameKind` per id, `KIND_UNCOMPUTED`
    /// where not yet probed) and `tokensByExprId_` (one canonical
    /// `SpecialTokenScan` blob per full-expression id).
    ///
    /// Entries are filled on first probe. Because filling mutates the memo,
    /// probes may only run in the prover's single-threaded phases (pre-burst
    /// setup and post-burst `standardProcessing`) — never from phase-2 parallel
    /// burst parts ([I-83](../../docs/agentic_swdd/30_invariants.md#i-83)).
    ///
    /// @invariant Cached values are pure functions of `nameMap.decode(id)`; scope
    ///            teardown (`wipeSubtree`) does not touch the memo, and CE clone
    ///            LBs carry valid copies. Single-threaded probe access only.
    /// @see classifyName, scanSpecialTokens, `TypedColdBlobMap`.
    struct EqClassNameCaches {
        /// @brief Sentinel marking an id whose kind has not been computed yet.
        static constexpr uint8_t KIND_UNCOMPUTED = 255;

        /// @brief `NameKind` per NameMap id, `KIND_UNCOMPUTED` where not yet
        ///        probed — a paged column on the LB's deloadable arena.
        PagedVector<uint8_t> kindById_;
        /// @brief Token scan per full-expression NameMap id — one canonical
        ///        `SpecialTokenScan` blob per id.
        TypedColdBlobMap<NameId, SpecialTokenScan> tokensByExprId_;
        TypedColdBlobMap<NameId, SpecialTokenScan>::KeysView tokensKeys;
        TypedColdBlobMap<NameId, SpecialTokenScan>::RunStartsView tokensRunStarts;
        TypedColdBlobMap<NameId, SpecialTokenScan>::BlobStartsView tokensBlobStarts;
        TypedColdBlobMap<NameId, SpecialTokenScan>::BlobPoolView tokensBlobPool;

        /// @brief Bind the paged columns + the token-map facets to the LB's
        ///        deloadable arena + dirty.
        ///
        /// @param arena The LB's deloadable per-LB `LbArena` (`LbMemory::manager`).
        /// @param dirty The LB's deload-dirty flag (`LbMemory::dirty`).
        EqClassNameCaches(LbArena* arena, DirtyState* dirty)
            : kindById_(arena, dirty),
              tokensByExprId_(arena, dirty),
              tokensKeys(&tokensByExprId_.inner()),
              tokensRunStarts(&tokensByExprId_.inner()),
              tokensBlobStarts(&tokensByExprId_.inner()),
              tokensBlobPool(&tokensByExprId_.inner()) {}
        EqClassNameCaches(const EqClassNameCaches&) = delete;
        EqClassNameCaches& operator=(const EqClassNameCaches&) = delete;

        /// @brief Memoized `classifyName` of `nm.decode(id)`.
        ///
        /// @details Defined out-of-line in `memory.hpp` (uses `classifyName` +
        /// `NameMap`). Grows `kindById_` on demand (one slot per id, paged);
        /// computes on first probe, then serves the stored tier.
        ///
        /// @param id NameMap id of a bare variable name; must be a real id (`>0`).
        /// @param nm The owning LB's NameMap (read-only; no minting).
        /// @return The name's tier.
        NameKind kindOf(NameId id, const NameMap& nm);

        /// @brief Memoized token scan of `nm.decode(exprId)` — zero-heap on
        ///        hit and miss, returned as a zero-copy VIEW over the cold
        ///        blob.
        ///
        /// @details Defined out-of-line in `memory.hpp` (uses `NameMap`, the
        /// occurrence scanners, and `SpecialTokenScanView`; full contract on
        /// the definition). One scan per distinct expression id: a hit peeks
        /// the stored record in place; a miss builds the canonical blob
        /// directly on @p peekArena via the regex-free occurrence scanners
        /// (byte-identical to the retired `scanSpecialTokens` + codec path),
        /// installs it, and re-peeks the installed record. The caller owns
        /// the arena window (no internal rewind) and consumes the view
        /// before its next `tokensByExprId_` mutation / arena pop. Filling
        /// mutates the memo — per-LB-exclusive phases only, never phase-2
        /// split parts.
        ///
        /// @param exprId    NameMap id of a full expression string; `>0`.
        /// @param nm        The owning LB's NameMap (read-only; no minting).
        /// @param peekArena Byte-bump arena for straddle copies + the miss
        ///                  build; caller pops after consuming the view.
        /// @return View over the memoized canonical blob.
        /// @invariant See `I-134`
        ///            (`docs/agentic_swdd/30_invariants.md`).
        SpecialTokenScanView tokensViewOf(NameId exprId, const NameMap& nm,
                                          ScratchArena& peekArena);

        /// @brief Drop the columns' page capacity — the grid-teardown release.
        void release() {
            kindById_.release();
            tokensByExprId_.release();
        }

        /// @brief Approximate live byte footprint across the two columns.
        /// @return Live bytes the memo occupies on the arena.
        int64_t liveBytes() const {
            return kindById_.liveBytes() + tokensByExprId_.liveBytes();
        }

        /// @brief Enumerate the deload facets at `base + 0..4` — the seam
        ///        `LbMemory::visitContainers` splices the instance in.
        ///
        /// @details `kindById_` is a paged column visited directly (`base+0`);
        /// `tokensByExprId_` emits 4 facets (key + run-starts + blob-starts +
        /// blob-pool) at `base+1..4`, key-facet-first.
        ///
        /// @tparam Self    `EqClassNameCaches` or `const EqClassNameCaches`.
        /// @tparam Visitor Callable `(uint32_t tag, facet&)`.
        /// @param self  The instance being enumerated.
        /// @param base  The instance's reserved tag-block base.
        /// @param visit The per-facet visitor.
        template <typename Self, typename Visitor>
        static void visitContainersImpl(Self& self, uint32_t base,
            Visitor&& visit) {
            visit(base + 0u, self.kindById_);
            visit(base + 1u, self.tokensKeys);
            visit(base + 2u, self.tokensRunStarts);
            visit(base + 3u, self.tokensBlobStarts);
            visit(base + 4u, self.tokensBlobPool);
        }

        /// @brief Mutable facet enumeration (load / release / discharge walk).
        /// @tparam Visitor Callable `(uint32_t tag, facet&)`.
        /// @param base  The instance's reserved tag-block base.
        /// @param visit The per-facet visitor.
        template <typename Visitor>
        void visitContainers(uint32_t base, Visitor&& visit) {
            visitContainersImpl(*this, base, std::forward<Visitor>(visit));
        }

        /// @brief Const facet enumeration (the deload DUMP reads a const LB).
        /// @tparam Visitor Callable `(uint32_t tag, const facet&)`.
        /// @param base  The instance's reserved tag-block base.
        /// @param visit The per-facet visitor.
        template <typename Visitor>
        void visitContainers(uint32_t base, Visitor&& visit) const {
            visitContainersImpl(*this, base, std::forward<Visitor>(visit));
        }
    };

} // namespace gl
