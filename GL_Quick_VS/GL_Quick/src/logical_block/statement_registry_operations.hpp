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
#include "statement_arrivals.hpp"

namespace gl {

/// @brief Updates shared statement storage without invoking proof-processing algorithms.
/// @details Storage, allocation and invocation order remain with callers.
class StatementRegistryOperations {
public:

/// @brief Publish a known-row mutation with its OR, implication-gate and
        ///        contradiction-demand wakeups.
        /// @details Used where the caller has already installed levels and does
        /// not also call stageWatchArrival. Keeping the notifications beside the
        /// write prevents an equality or derived-row path from losing its event.
        /// @param mb Owning block with the caller's finalized level row.
        /// @param key Original expression and validity.
        /// @param local Ownership for a fresh row; existing rows keep their first value.
        /// @return Void.
static void publishKnownStatementKey(Memory& mb, int64_t key, bool local);

/// @brief Remove a known row and stage its OR membership reconciliation.
        /// @details A missing row remains the defined idempotent removal case.
        /// The caller owns level and ordinary negation-watch cleanup; the OR
        /// notification stays in the same mutation order as the table erasure.
        /// @param mb Owning block.
        /// @param original Original expression identity.
        /// @param scope Source-validity identity.
        /// @return Void.
static void eraseKnownStatementKey(Memory& mb, NameId original, NameId scope);

/// @brief `StrSpan` door of @ref removeExpressionFromMemoryBlock — erase a
        ///        statement (`state==0`) or discharge a goal (`state==1`) addressed
        ///        by its `(original, validityName)` bytes, with no
        ///        `EncodedExpression` materialized.
        ///
        /// @details
        /// Byte-for-byte the same procedure as the `EncodedExpression` overload,
        /// which now delegates here. Both names are resolved through the
        /// non-minting `NameMap::lookup(StrSpan)` — identical verdict to
        /// `lookup(const std::string&)` (which itself forwards to the span form) —
        /// so a never-interned target finds no stored row and the sweep is the same
        /// no-op the string scan produced. `state == 0` erases every matching
        /// `(originalId, validityId)` row from `intLocalEncodedStatements` /
        /// `intLocalEncodedStatementsDelta` / `intEncodedStatements` (back to front,
        /// indices stable) and repairs the persistent per-class statement-index
        /// waterlines for the `intEncodedStatements` positions it erased (see
        /// `repairEqClassWaterlines` — a stale waterline silently under-applies
        /// classes); `state == 1` discharges the goal from `intToBeProved`
        /// (the caller guarantees both names interned, so a miss is a contract
        /// violation asserted, not a probe miss). Lets a span-holding caller —
        /// `ordisMerge`'s branch-cleanup loop, which builds each branch validity on
        /// the scratch arena — remove without a transient `EncodedExpression`.
        ///
        /// @param original     The statement's canonical expression bytes.
        /// @param validityName The statement's validity-scope bytes.
        /// @param mb           The owning memory block.
        /// @param state        `0` = erase encoded-statement rows; `1` = discharge
        ///                     the `intToBeProved` goal.
        /// @see removeExpressionFromMemoryBlock(const EncodedExpression&, Memory&, int)
        ///      — the delegating owning-string overload.
/// @return Nothing; updates the specified storage in place.
static void removeExpressionFromMemoryBlock(StrSpan original, StrSpan validityName, Memory& mb, int state);

/// @brief Remove encoded rows or a goal addressed by an owning expression.
/// @details Delegates to the span overload; the caller chooses the same storage
/// operation and retains ownership of both expression strings.
/// @param encExpr Expression and exact scope to remove.
/// @param mb Owning resident block.
/// @param state Zero removes statement-list rows; one removes the existing goal.
/// @return Nothing; updates the specified storage in place.
static void removeExpressionFromMemoryBlock(const EncodedExpression& encExpr, Memory& mb, int state);

/// @brief Statement-row removals collected over one drain and applied in one
///        pass over the three statement lists.
///
/// @details
/// `removeExpressionFromMemoryBlock(state 0)` scans all three statement lists
/// for every removal. A drain that converges several OR cohorts requests
/// several removals in a row while nothing reads those lists or the class
/// waterlines, so the batch records each statement's identifier pair when its
/// removal is requested and erases every recorded pair at once: one stable
/// forward compaction per list (`SharedExpressionList::eraseIf`) and one
/// waterline repair over all erased `intEncodedStatements` positions. The
/// surviving rows, their order and the waterlines equal those of the removals
/// made one by one; only the recycling order of shared record slots differs,
/// and no reader observes it (the canonical stream expands records in list
/// order). The recorded pairs ride the page tier of a gen-scratch arena, so
/// the caller's byte-tier scratch scopes may rewind between requests.
/// Single-threaded per logic block (I-83).
///
/// @invariant Between the first `stage` and `apply` nothing reads or writes
///            the three statement lists or `eqClassSttmntIndexMapMap`; the
///            owner limits the batch to such a window.
/// @see removeExpressionFromMemoryBlock, repairEqClassWaterlines,
///      CaseProcessor::ordisMerge, EqualityPropagation::drainStagedEquiProducts.
class RowRemovalBatch {
public:
    /// @brief Bind an empty batch to a gen-scratch arena.
    ///
    /// @details
    /// Nothing is allocated until the first staged removal. The arena must be
    /// a per-slot `genScratchArenas()` arena (page tier) that outlives the
    /// batch.
    ///
    /// @param arena The gen-scratch arena holding the recorded pairs.
    explicit RowRemovalBatch(ScratchArena& arena);
    RowRemovalBatch(const RowRemovalBatch&) = delete;
    RowRemovalBatch& operator=(const RowRemovalBatch&) = delete;

    /// @brief Record the removal of every row of one statement.
    ///
    /// @details
    /// Resolves both names through the non-minting `NameMap::lookup`, exactly
    /// as the immediate removal does. A name that was never interned has no
    /// stored row, so nothing is recorded (the immediate removal's no-op).
    /// The same pair may be recorded more than once; it is removed once.
    ///
    /// @param original     The statement's canonical expression bytes.
    /// @param validityName The statement's validity-scope bytes.
    /// @param mb           The owning memory block.
    /// @return Nothing.
    void stage(StrSpan original, StrSpan validityName, const Memory& mb);

    /// @brief Erase the rows of every recorded statement and repair the
    ///        waterlines, then empty the batch.
    ///
    /// @details
    /// Sorts the recorded pairs (duplicates dropped), compacts
    /// `intLocalEncodedStatements` and `intLocalEncodedStatementsDelta`, then
    /// `intEncodedStatements` while collecting its erased positions in
    /// ascending order, and repairs `eqClassSttmntIndexMapMap` once over all of
    /// them. The per-statement duplicate tripwire of the immediate removal
    /// (at most eight rows of one pair) is kept. An empty batch changes
    /// nothing.
    ///
    /// @param mb The owning memory block; the same block every pair was
    ///           recorded against.
    /// @return Nothing.
    /// @invariant The lists and waterlines equal those left by
    ///            `removeExpressionFromMemoryBlock(state 0)` called once per
    ///            recorded pair, in any order.
    void apply(Memory& mb);

private:
    ScratchArena& arena_;
    DirtyState dirty_ = DirtyState::Clean;
    PagedVector<int64_t> keys_;
};

/// @brief Repair the persistent per-class statement-index waterlines
    ///        after rows were erased from `intEncodedStatements`.
    ///
    /// @details
    /// `eqClassSttmntIndexMapMap` stores, per (validity, class-member-set)
    /// key, the `intEncodedStatements` position from which the next
    /// class-application pass resumes. Erasing a row at position `p`
    /// shifts every later row down by one, so a stored waterline greater
    /// than `p` points one row too far and the pass silently skips the
    /// row that slid underneath — an under-application with no assert.
    /// This helper decrements each stored waterline by the number of
    /// erased positions strictly below it, restoring the exact logical
    /// resume point. The erased run is strictly ASCENDING (asserted), so
    /// the count below a waterline is one binary search — a cleanup
    /// compaction can erase thousands of rows while the map holds
    /// thousands of keys, and a linear count per key would make the
    /// repair quadratic. Walks the map by id (the adjustment is
    /// order-free — no observable iteration order, I-84 concerns none);
    /// in-place `setValueAt` on the inner map (forces `Restructured`,
    /// I-119).
    ///
    /// @tparam Run    Indexable run of `int32_t` positions — a C array /
    ///                `const int32_t*` or a `PagedVector<int32_t>`; only
    ///                `operator[]` is read.
    /// @param mb      Owning LB (`eqClassSttmntIndexMapMap` mutated in
    ///                place).
    /// @param erased  Positions (pre-erase indices) removed from
    ///                `intEncodedStatements`, strictly ascending.
    /// @param erasedN Number of positions in @p erased (> 0).
    /// @invariant Every consumer of a waterline resumes at the same
    ///            logical row before and after a removal + repair pair.
    /// @see removeExpressionFromMemoryBlock and cleanUpExpressions — the
    ///      two erasing callers; lookupEqClassIndex / upsertEqClassIndex —
    ///      the waterline read/write doors.
/// @return Nothing; updates the specified storage in place.
template <class Run>
    static inline void repairEqClassWaterlines(Memory& mb, const Run& erased,
                                        int32_t erasedN);
};

/// @brief Publish a known-row mutation with its OR, implication-gate and
        ///        contradiction-demand wakeups.
        /// @details Used where the caller has already installed levels and does
        /// not also call stageWatchArrival. Keeping the notifications beside the
        /// write prevents an equality or derived-row path from losing its event.
        /// @param mb Owning block with the caller's finalized level row.
        /// @param key Original expression and validity.
        /// @param local Ownership for a fresh row; existing rows keep their first value.
        /// @return Void.
inline void StatementRegistryOperations::publishKnownStatementKey(Memory& mb, int64_t key, bool local) {
            upsertStatementKey(mb.intKnownStatements, key, local);
            StatementArrivals::stageOrFactChange(mb, key);
            StatementArrivals::stageSubproofArrival(mb, key);
            StatementArrivals::stageContradictionArrival(mb, key);
        }

/// @brief Remove a known row and stage its OR membership reconciliation.
        /// @details A missing row remains the defined idempotent removal case.
        /// The caller owns level and ordinary negation-watch cleanup; the OR
        /// notification stays in the same mutation order as the table erasure.
        /// @param mb Owning block.
        /// @param original Original expression identity.
        /// @param scope Source-validity identity.
        /// @return Void.
inline void StatementRegistryOperations::eraseKnownStatementKey(Memory& mb, NameId original, NameId scope) {
            mb.intKnownStatements.erase(StatementKey{original, scope});
            StatementArrivals::stageOrFactChange(mb, packStatementKey(original, scope));
        }

/// @brief Repair the persistent per-class statement-index waterlines
    ///        after rows were erased from `intEncodedStatements`.
    ///
    /// @details
    /// `eqClassSttmntIndexMapMap` stores, per (validity, class-member-set)
    /// key, the `intEncodedStatements` position from which the next
    /// class-application pass resumes. Erasing a row at position `p`
    /// shifts every later row down by one, so a stored waterline greater
    /// than `p` points one row too far and the pass silently skips the
    /// row that slid underneath — an under-application with no assert.
    /// This helper decrements each stored waterline by the number of
    /// erased positions strictly below it, restoring the exact logical
    /// resume point. The erased run is strictly ASCENDING (asserted), so
    /// the count below a waterline is one binary search — a cleanup
    /// compaction can erase thousands of rows while the map holds
    /// thousands of keys, and a linear count per key would make the
    /// repair quadratic. Walks the map by id (the adjustment is
    /// order-free — no observable iteration order, I-84 concerns none);
    /// in-place `setValueAt` on the inner map (forces `Restructured`,
    /// I-119).
    ///
    /// @tparam Run    Indexable run of `int32_t` positions — a C array /
    ///                `const int32_t*` or a `PagedVector<int32_t>`; only
    ///                `operator[]` is read.
    /// @param mb      Owning LB (`eqClassSttmntIndexMapMap` mutated in
    ///                place).
    /// @param erased  Positions (pre-erase indices) removed from
    ///                `intEncodedStatements`, strictly ascending.
    /// @param erasedN Number of positions in @p erased (> 0).
    /// @invariant Every consumer of a waterline resumes at the same
    ///            logical row before and after a removal + repair pair.
    /// @see removeExpressionFromMemoryBlock and cleanUpExpressions — the
    ///      two erasing callers; lookupEqClassIndex / upsertEqClassIndex —
    ///      the waterline read/write doors.
/// @return Nothing; updates the specified storage in place.
template <class Run>
    inline void StatementRegistryOperations::repairEqClassWaterlines(Memory& mb, const Run& erased,
                                        int32_t erasedN) {
        assert(erasedN > 0 && "repairEqClassWaterlines: nothing erased");
        for (int32_t k = 1; k < erasedN; ++k) {
            assert(erased[k - 1] < erased[k]
                && "repairEqClassWaterlines: erased run must be strictly ascending");
        }
        const int32_t n = mb.eqClassSttmntIndexMapMap.count();
        for (int32_t id = 1; id <= n; ++id) {
            const int wl = mb.eqClassSttmntIndexMapMap.valueAt(id);
            // Count of erased positions strictly below wl == the index of
            // the first position >= wl in the ascending run.
            int32_t lo = 0;
            int32_t hi = erasedN;
            while (lo < hi) {
                const int32_t mid = lo + (hi - lo) / 2;
                if (erased[mid] < wl) lo = mid + 1; else hi = mid;
            }
            if (lo > 0) {
                mb.eqClassSttmntIndexMapMap.inner().setValueAt(id, wl - lo);
            }
        }
    }
}
